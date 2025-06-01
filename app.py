from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
import os
import spacy
import pdfplumber
import PyPDF2
from transformers import pipeline
from werkzeug.utils import secure_filename
import logging
import time
import razorpay
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import sqlalchemy.exc
from sqlalchemy.sql import text  # For raw SQL queries
import re

# Setup logging with more detail
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('pdfplumber').setLevel(logging.ERROR)  # Suppress pdfplumber warnings

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Set SECRET_KEY from environment variable
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    logger.error("No SECRET_KEY set. Please set the SECRET_KEY environment variable.")
    raise ValueError("No SECRET_KEY set for Flask application. Set the SECRET_KEY environment variable in Render or .env file.")

# Configuration
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
FREE_WORD_LIMIT = 500  # Word limit for free trial
FREE_UPLOAD_LIMIT = 5  # Max free uploads

# Ensure instance directory exists
app.instance_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'instance'))
logger.debug(f"Instance path: {app.instance_path}")
try:
    os.makedirs(app.instance_path, exist_ok=True)
    logger.debug(f"Created instance directory: {app.instance_path}")
except Exception as e:
    logger.error(f"Failed to create instance directory: {str(e)}")
    raise

UPLOAD_FOLDER = os.path.join(app.instance_path, 'Uploads')
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.debug(f"Created uploads directory: {UPLOAD_FOLDER}")
except Exception as e:
    logger.error(f"Failed to create uploads directory: {str(e)}")
    raise
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Database configuration with error handling
# Database configuration with error handling
database_url = os.getenv('DATABASE_URL')
logger.debug(f"Raw DATABASE_URL from environment: {database_url}")
if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
else:
    # Local development fallback (cross-platform)
    db_dir = os.path.join(app.instance_path, 'db')
    try:
        os.makedirs(db_dir, exist_ok=True)
        logger.debug(f"Created database directory: {db_dir}")
    except Exception as e:
        logger.error(f"Failed to create database directory: {str(e)}")
        raise
    db_path = os.path.join(db_dir, 'users.db')
    logger.info(f"Database path: {db_path}")
    database_url = f"sqlite:///{db_path}"
logger.debug(f"Configured database URL: {database_url}")
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Initialize SQLAlchemy, Migrate, and Bcrypt
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

# Test database connection
try:
    with app.app_context():
        db.session.execute(text('SELECT 1'))  # Use text() for raw SQL
        logger.info("Database connection successful")
except sqlalchemy.exc.OperationalError as e:
    logger.error(f"Database connection failed: {str(e)}")
    raise RuntimeError("Failed to connect to the database. Check DATABASE_URL and database availability.")

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Razorpay configuration with debug logging
try:
    razorpay_key_id = os.getenv('RAZORPAY_KEY_ID')
    razorpay_key_secret = os.getenv('RAZORPAY_KEY_SECRET')
    logger.debug(f"Razorpay Key ID: {razorpay_key_id}")
    logger.debug(f"Razorpay Key Secret: {razorpay_key_secret[:4] if razorpay_key_secret else None}...")
    if not razorpay_key_id or not razorpay_key_secret:
        logger.error("Razorpay credentials missing. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET environment variables.")
        razorpay_client = None
    else:
        razorpay_client = razorpay.Client(auth=(razorpay_key_id, razorpay_key_secret))
        razorpay_client.set_app_details({"title": "CaseSummarizer", "version": "1.0"})
        logger.debug("Razorpay client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Razorpay client: {str(e)}")
    razorpay_client = None

# Global model instances (load on-demand to save memory)
summarizer = None
nlp = None

# User model with subscription
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    razorpay_customer_id = db.Column(db.String(100))
    subscription_status = db.Column(db.String(20), default='inactive')
    subscription_id = db.Column(db.String(100))
    upload_count = db.Column(db.Integer, default=0)
    subscription_end_date = db.Column(db.DateTime)

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Loading summarization model...")
        try:
            # Use google/pegasus-xsum (Apache 2.0 licensed, suitable for legal text summarization)
            summarizer = pipeline(
                "summarization",
                model="google/pegasus-xsum",
                framework="pt",
                device=-1
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {str(e)}", exc_info=True)
            raise
    return summarizer

def load_nlp():
    global nlp
    if nlp is None:
        logger.info("Loading NLP model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("NLP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP model: {str(e)}", exc_info=True)
            raise
    return nlp

def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        start_time = time.time()
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            logger.warning("pdfplumber extracted no text, falling back to PyPDF2")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        if not text.strip():
            return "Error: PDF contains no extractable text"
        
        logger.info(f"Text extraction completed in {time.time() - start_time:.2f}s, length: {len(text)} characters")
        return text
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def preprocess_legal_text(text):
    try:
        # Clean up text and remove noise specific to legal documents
        text = ' '.join(text.split())
        noise_phrases = [
            "IN THE COURT OF", "CASE NO.", "JUDGMENT", 
            "BEFORE THE HON'BLE", "IN THE MATTER OF",
            "BEFORE THE HONOURABLE", "REPORTABLE", "NON-REPORTABLE",
            "WHEREAS", "PURSUANT TO", "ORDERED AND ADJUDGED"
        ]
        for phrase in noise_phrases:
            text = text.replace(phrase, "")
        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return text

def summarize_legal_text(text):
    try:
        logger.info("Starting legal text summarization...")
        start_time = time.time()
        
        summarizer = load_summarizer()
        text = preprocess_legal_text(text)
        
        # Check word count for free trial
        word_count = len(text.split())
        if current_user.subscription_status != 'active':
            if word_count > FREE_WORD_LIMIT:
                return f"Error: Free trial limited to {FREE_WORD_LIMIT} words. Please subscribe for unlimited processing."

        # Adjust chunking for legal text (focus on sections like holdings, arguments)
        if current_user.subscription_status == 'active':
            max_length = 10000
            chunk_size = 1000
            max_chunks = 10
        else:
            max_length = 5000
            chunk_size = 200
            max_chunks = 5
        
        text = text[:max_length]
        # Split into meaningful legal sections (e.g., by paragraphs or headings)
        sections = re.split(r'\n\s*\n', text)  # Split by double newlines (paragraphs)
        chunks = []
        current_chunk = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(current_chunk) + len(section) <= chunk_size:
                current_chunk += " " + section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
            if len(chunks) >= max_chunks:
                break
        if current_chunk and len(chunks) < max_chunks:
            chunks.append(current_chunk.strip())
        
        if not chunks:
            return "Error: No valid sections found for summarization"
        
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                # Summarize with a focus on legal outcomes and arguments
                summary = summarizer(
                    chunk,
                    max_length=50,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                # Post-process to prioritize legal terms
                summary = re.sub(r'\b(the|is|are)\b', '', summary)  # Remove filler words
                summaries.append(summary.strip())
                logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.warning(f"Error summarizing chunk {i+1}: {str(e)}", exc_info=True)
                continue
        
        if not summaries:
            return "Error: Could not generate any summary chunks"
        
        full_summary = " ".join(summaries)
        logger.info(f"Summarization completed in {time.time() - start_time:.2f}s")
        return full_summary
    except Exception as e:
        error_msg = f"Error summarizing text: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def extract_legal_entities(text):
    try:
        logger.info("Extracting legal entities...")
        start_time = time.time()
        
        nlp = load_nlp()
        
        char_limit = 25000 if current_user.subscription_status == 'active' else 10000
        text = text[:char_limit]
        
        doc = nlp(text)
        
        legal_tags = {
            "Parties": set(),
            "Judges": set(),
            "Courts": set(),
            "Dates": set(),
            "Laws": set(),
            "Case Numbers": set(),
            "Citations": set()
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if "J." in ent.text or "Justice" in ent.text:
                    legal_tags["Judges"].add(ent.text)
                else:
                    legal_tags["Parties"].add(ent.text)
            elif ent.label_ == "ORG":
                if "court" in ent.text.lower() or "high court" in ent.text.lower():
                    legal_tags["Courts"].add(ent.text)
            elif ent.label_ == "DATE":
                legal_tags["Dates"].add(ent.text)
            elif ent.label_ == "LAW":
                legal_tags["Laws"].add(ent.text)
            elif ent.label_ == "CARDINAL":
                if any(word in ent.text.lower() for word in ["no.", "number"]):
                    legal_tags["Case Numbers"].add(ent.text)
        
        for sent in doc.sents:
            if " v. " in sent.text:
                parts = [p.strip() for p in sent.text.split(" v. ")]
                if len(parts) == 2:
                    legal_tags["Parties"].update(parts)
            if any(c in sent.text for c in [" U.S. ", " F. ", " S.Ct. ", " A.C. "]):
                legal_tags["Citations"].add(sent.text.strip())
        
        result = {k: sorted(v) for k, v in legal_tags.items() if v}
        logger.info(f"Entity extraction completed in {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        error_msg = f"Error extracting entities: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"Error": error_msg}

# Initialize database tables at startup
with app.app_context():
    try:
        logger.info("Successfully created database tables")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def index():
    try:
        logger.debug("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}", exc_info=True)
        return render_template('error.html', message="Internal Server Error"), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            
            if not username or not password:
                flash('Username and password are required', 'error')
                return redirect(url_for('login'))
            
            logger.debug(f"Login attempt for username: {username}")
            user = db.session.get(User, User.query.filter_by(username=username).first().id) if User.query.filter_by(username=username).first() else None
            
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                logger.info(f"User {username} logged in successfully")
                flash('Logged in successfully', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Login Error"), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            logger.debug(f"Received registration form data: {request.form}")
            
            if not username or not password:
                flash('Username and password are required', 'error')
                return redirect(url_for('register'))
            
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('register'))
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return redirect(url_for('register'))
            
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(
                username=username,
                password=hashed_password,
                subscription_status='inactive',
                upload_count=0
            )
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful, please log in', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return render_template('error.html', message=f"Registration Error: {str(e)}"), 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        flash('Logged out successfully', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in logout route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Logout Error"), 500

@app.route('/create-subscription', methods=['GET'])
@login_required
def create_subscription():
    try:
        if not razorpay_client:
            flash('Payment system is currently unavailable', 'error')
            return redirect(url_for('index'))
        
        if not current_user.razorpay_customer_id:
            try:
                customer = razorpay_client.customer.create({
                    'name': current_user.username,
                    'email': f"{current_user.username}@example.com",  # Placeholder email
                    'contact': '9000000000'
                })
                current_user.razorpay_customer_id = customer['id']
                db.session.commit()
            except Exception as e:
                logger.error(f"Error creating customer: {str(e)}", exc_info=True)
                flash('Error creating customer record', 'error')
                return redirect(url_for('index'))
        
        try:
            # Create a one-month subscription order (not a recurring subscription)
            amount = 9900  # ₹99 in paise
            order = razorpay_client.order.create({
                'amount': amount,
                'currency': 'INR',
                'payment_capture': 1,
                'notes': {
                    'user_id': current_user.id,
                    'username': current_user.username
                }
            })
            
            return render_template('payment.html', 
                                 order_id=order['id'],
                                 key_id=os.getenv('RAZORPAY_KEY_ID'),
                                 username=current_user.username)
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}", exc_info=True)
            flash('Error creating subscription order', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Unexpected error in create-subscription: {str(e)}", exc_info=True)
        return render_template('error.html', message="Subscription Error"), 500

@app.route('/payment-success', methods=['POST'])
@login_required
def payment_success():
    try:
        order_id = request.form.get('razorpay_order_id')
        razorpay_payment_id = request.form.get('razorpay_payment_id')
        razorpay_signature = request.form.get('razorpay_signature')
        
        if not all([order_id, razorpay_payment_id, razorpay_signature]):
            flash('Invalid payment response', 'error')
            return redirect(url_for('index'))
        
        params = {
            'razorpay_order_id': order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        try:
            # Verify the payment signature
            razorpay_client.utility.verify_payment_signature(params)
            payment = razorpay_client.payment.fetch(razorpay_payment_id)
            
            if payment['status'] == 'captured':
                current_user.subscription_status = 'active'
                current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
                db.session.commit()
                flash('Subscription successful! Valid for 30 days.', 'success')
            else:
                flash('Payment not captured', 'warning')
        except Exception as e:
            logger.error(f"Payment verification failed: {str(e)}", exc_info=True)
            flash('Payment verification failed', 'error')
        
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in payment-success route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Payment Error"), 500

@app.route('/payment-failure', methods=['GET', 'POST'])
@login_required
def payment_failure():
    try:
        if request.method == 'POST':
            error_code = request.form.get('error_code')
            error_description = request.form.get('error_description')
            logger.error(f"Payment failed: {error_code} - {error_description}")
        else:
            logger.error("Payment failure accessed via GET")
        
        flash('Payment failed. Please try again.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in payment-failure route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Payment Error"), 500

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    try:
        if request.method == 'GET':
            return render_template('upload.html')
        
        # Check subscription status
        if current_user.subscription_status != 'active':
            if current_user.upload_count >= FREE_UPLOAD_LIMIT:
                flash('Free trial limit reached. Subscribe for unlimited uploads.', 'error')
                return redirect(url_for('create_subscription'))
            if current_user.subscription_end_date and current_user.subscription_end_date < datetime.utcnow():
                current_user.subscription_status = 'inactive'
                db.session.commit()
                flash('Your subscription has expired. Please renew.', 'error')
                return redirect(url_for('create_subscription'))
        
        if 'file' not in request.files:
            flash('No file part in the request', 'error')
            return redirect(url_for('upload_file'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('upload_file'))
        
        if not allowed_file(file.filename):
            flash('Only PDF files are allowed', 'error')
            return redirect(url_for('upload_file'))
        
        filename = secure_filename(file.filename)
        unique_id = str(int(time.time()))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
        try:
            file.save(filepath)
            logger.info(f"File saved temporarily at: {filepath}")
            
            file_size = os.path.getsize(filepath)
            if file_size > 20 * 1024 * 1024:  # 20MB
                flash('File is too large to process (max 20MB)', 'error')
                os.remove(filepath)
                return redirect(url_for('upload_file'))
        except Exception as e:
            flash('Failed to save file', 'error')
            logger.error(f"File save error: {str(e)}", exc_info=True)
            return redirect(url_for('upload_file'))
        
        try:
            text = extract_text_from_pdf(filepath)
            if text.startswith("Error"):
                flash(text, 'error')
                return redirect(url_for('upload_file'))
            
            summary = summarize_legal_text(text)
            if summary.startswith("Error"):
                flash(summary, 'error')
                return redirect(url_for('upload_file'))
            
            tags = extract_legal_entities(text)
            
            if current_user.subscription_status != 'active':
                current_user.upload_count += 1
                db.session.commit()
            
            return render_template(
                'result.html',
                summary=summary,
                tags=tags,
                original_filename=filename,
                text_preview=text[:500] + "..." if len(text) > 500 else text,
                is_premium=current_user.subscription_status == 'active'
            )
        except Exception as e:
            flash(f"Processing error: {str(e)}", 'error')
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return redirect(url_for('upload_file'))
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Temporary file removed: {filepath}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}", exc_info=True)
        return render_template('error.html', message="Upload Error"), 500

@app.route('/account')
@login_required
def account():
    try:
        return render_template('account.html', 
                            user=current_user,
                            is_premium=current_user.subscription_status == 'active')
    except Exception as e:
        logger.error(f"Error in account route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Account Error"), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)