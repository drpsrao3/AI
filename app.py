from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
import os
import spacy
import pdfplumber
import PyPDF2
from transformers import pipeline, BartTokenizer
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
logging.getLogger('transformers').setLevel(logging.ERROR)  # Suppress transformers warnings

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
FREE_WORD_LIMIT = 1500  # Word limit for free trial
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

# Replace the database configuration section in app.py
database_url = os.getenv('DATABASE_URL')
logger.debug(f"Raw DATABASE_URL from environment: {'[hidden]' if os.getenv('RENDER') else database_url}")
if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    # Ensure sslmode=require and add SSL root certificate
    if 'sslmode' not in database_url:
        database_url += '?sslmode=require'
    # Optional: Specify SSL root certificate (Render may require this)
    if os.getenv('RENDER'):
        database_url += '&sslrootcert=/etc/ssl/certs/ca-certificates.crt'
else:
    if os.getenv('RENDER'):
        logger.error("DATABASE_URL environment variable is not set on Render. This is required.")
        raise ValueError("DATABASE_URL environment variable is not set on Render. This is required.")
    db_dir = os.path.join(app.instance_path, 'db')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'users.db')
    logger.info(f"Local database path (SQLite): {db_path}")
    database_url = f"sqlite:///{db_path}"
logger.debug(f"Configured database URL: {'[hidden]' if os.getenv('RENDER') else database_url}")
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_timeout': 10,
    'pool_recycle': 300,
    'connect_args': {'sslmode': 'require', 'sslrootcert': '/etc/ssl/certs/ca-certificates.crt' if os.getenv('RENDER') else None}
}

# Initialize SQLAlchemy, Migrate, and Bcrypt
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

# Test database connection with retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def test_db_connection():
    with app.app_context():
        db.session.execute(text('SELECT 1'))
        logger.info("Database connection successful")

try:
    test_db_connection()
except sqlalchemy.exc.OperationalError as e:
    logger.error(f"Database connection failed: {str(e)}")
    raise RuntimeError(f"Failed to connect to the database: {str(e)}. Check DATABASE_URL and database availability.")

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
tokenizer = None  # Add tokenizer for BART
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
            # Use sshleifer/distilbart-cnn-12-6 for better handling of legal texts
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                framework="pt",
                device=-1
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {str(e)}", exc_info=True)
            raise
    return summarizer

def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        logger.info("Loading BART tokenizer...")
        try:
            tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            logger.info("BART tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}", exc_info=True)
            raise
    return tokenizer

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
        # Clean up text and remove noise specific to Indian legal documents
        text = ' '.join(text.split())
        noise_phrases = [
            "IN THE COURT OF", "CASE NO.", "JUDGMENT", 
            "BEFORE THE HON'BLE", "IN THE MATTER OF",
            "BEFORE THE HONOURABLE", "REPORTABLE", "NON-REPORTABLE",
            "WHEREAS", "PURSUANT TO", "ORDERED AND ADJUDGED",
            "Writ Petition", "Civil Appeal", "Criminal Appeal",
            "Special Leave Petition", "SLP", "AIR \d{4} SC \d+",  # e.g., AIR 2017 SC 1234
            "Section \d+ of the",  # e.g., Section 377 of the
            "[A-Z]+\. \d{4}"  # e.g., SCC 2017
        ]
        for phrase in noise_phrases:
            text = re.sub(phrase, "", text, flags=re.IGNORECASE)
        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return text

def summarize_legal_text(text):
    try:
        logger.info("Starting legal text summarization...")
        start_time = time.time()
        
        summarizer = load_summarizer()
        tokenizer = load_tokenizer()
        text = preprocess_legal_text(text)
        
        # Check word count for free trial
        word_count = len(text.split())
        if current_user.subscription_status != 'active':
            if word_count > FREE_WORD_LIMIT:
                return f"Error: Free trial limited to {FREE_WORD_LIMIT} words. Please subscribe for unlimited processing."

        # Adjust chunking for legal text (focus on token count instead of character count)
        if current_user.subscription_status == 'active':
            max_length = 10000  # Character limit for the entire text
            max_tokens_per_chunk = 512  # Safe token limit per chunk (BART max is 1024, we use half to be safe)
            max_chunks = 10
            summary_max_length = 100  # Longer summaries for subscribed users
            summary_min_length = 50
        else:
            max_length = 5000
            max_tokens_per_chunk = 256  # Smaller chunks for free users
            max_chunks = 5
            summary_max_length = 50
            summary_min_length = 30
        
        text = text[:max_length]
        # Split into meaningful legal sections (e.g., by paragraphs or headings)
        sections = re.split(r'\n\s*\n', text)  # Split by double newlines (paragraphs)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            # Split long sections into smaller pieces if necessary
            section_words = section.split()
            temp_section = ""
            for word in section_words:
                temp_section += word + " "
                temp_tokens = len(tokenizer.encode(temp_section.strip(), add_special_tokens=False))
                if temp_tokens > max_tokens_per_chunk:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = temp_section.strip()
                    current_tokens = temp_tokens
                    temp_section = ""
                    if len(chunks) >= max_chunks:
                        break
            if temp_section:
                section_tokens = len(tokenizer.encode(temp_section.strip(), add_special_tokens=False))
                if current_tokens + section_tokens <= max_tokens_per_chunk:
                    current_chunk += " " + temp_section.strip()
                    current_tokens += section_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = temp_section.strip()
                    current_tokens = section_tokens
            if len(chunks) >= max_chunks:
                break
        if current_chunk and len(chunks) < max_chunks:
            chunks.append(current_chunk.strip())
        
        if not chunks:
            return "Error: No valid sections found for summarization"
        
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                # Manually truncate to 512 tokens to be extra safe
                encoded = tokenizer.encode(chunk, add_special_tokens=True, truncation=True, max_length=512)
                truncated_chunk = tokenizer.decode(encoded, skip_special_tokens=True)
                # Summarize with a focus on legal outcomes and arguments
                summary = summarizer(
                    truncated_chunk,
                    max_length=summary_max_length,
                    min_length=summary_min_length,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                # Post-process to prioritize legal outcomes and Indian judicial terms
                legal_keywords = ["held", "dismissed", "upheld", "granted", "denied", "observed", "ruled", "constitution", "ipc"]
                for keyword in legal_keywords:
                    if keyword in summary.lower():
                        # Prioritize sentences with legal outcomes
                        summary = f"{keyword.capitalize()} {summary.split(keyword, 1)[1].strip()}"
                        break
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
            "Citations": set(),
            "Indian Statutes": set()
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if "J." in ent.text or "Justice" in ent.text:
                    legal_tags["Judges"].add(ent.text)
                else:
                    legal_tags["Parties"].add(ent.text)
            elif ent.label_ == "ORG":
                if "court" in ent.text.lower() or "tribunal" in ent.text.lower():
                    legal_tags["Courts"].add(ent.text)
            elif ent.label_ == "DATE":
                legal_tags["Dates"].add(ent.text)
            elif ent.label_ == "LAW":
                legal_tags["Laws"].add(ent.text)
            elif ent.label_ == "CARDINAL":
                if any(word in ent.text.lower() for word in ["no.", "number"]):
                    legal_tags["Case Numbers"].add(ent.text)
        
        # Enhanced entity extraction for Indian judicial contexts
        for sent in doc.sents:
            sent_text = sent.text
            # Parties (e.g., "Shri Justice C.S. Karnan vs Registrar General")
            if " v. " in sent_text or " vs. " in sent_text:
                parts = [p.strip() for p in sent_text.replace(" vs. ", " v. ").split(" v. ")]
                if len(parts) == 2:
                    legal_tags["Parties"].update(parts)
            # Citations (e.g., AIR 2017 SC 1234, (2017) 2 SCC 123)
            if re.search(r"\b(AIR \d{4} SC \d+|SCC \d{4}\b|\(\d{4}\))", sent_text):
                legal_tags["Citations"].add(sent_text.strip())
            # Indian Statutes (e.g., Section 377 IPC, Article 21 of the Constitution)
            if re.search(r"(Section \d+ [A-Z]+|Article \d+ of the Constitution)", sent_text):
                legal_tags["Indian Statutes"].add(sent_text.strip())
            # Courts (e.g., Supreme Court of India, Madras High Court)
            if re.search(r"(Supreme Court of India|High Court of [A-Za-z]+)", sent_text):
                legal_tags["Courts"].add(sent_text.strip())
        
        result = {k: sorted(v) for k, v in legal_tags.items() if v}
        logger.info(f"Entity extraction completed in {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        error_msg = f"Error extracting entities: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"Error": error_msg}

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
            
            logger.debug(f"Received login request with username: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
            
            if not username or not password:
                logger.warning("Username or password missing in login request")
                flash('Username and password are required', 'error')
                return redirect(url_for('login'))
            
            logger.debug(f"Querying database for username: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
            user_query = User.query.filter_by(username=username).first()
            if not user_query:
                logger.warning(f"No user found with username: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
                flash('Invalid username or password', 'error')
                return redirect(url_for('login'))
            
            logger.debug(f"Fetching user with ID: {user_query.id}")
            user = db.session.get(User, user_query.id)
            if not user:
                logger.error(f"Failed to fetch user with ID: {user_query.id}")
                flash('Invalid username or password', 'error')
                return redirect(url_for('login'))
            
            logger.debug(f"Verifying password for user: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
            if bcrypt.check_password_hash(user.password, password):
                logger.debug(f"Password verified, logging in user: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
                login_user(user)
                logger.info(f"User {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'} logged in successfully")
                flash('Logged in successfully', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                logger.warning(f"Password verification failed for user: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
                flash('Invalid username or password', 'error')
                return redirect(url_for('login'))
        
        logger.debug("Rendering login page for GET request")
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
            
            logger.debug(f"Received registration form data with username: {username[:4] + '...' + username.split('@')[-1] if '@' in username else 'hidden'}")
            
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
                    'email': f"{current_user.username.split('@')[0]}@example.com",  # Strip existing domain and use a valid one
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
        user_data = {
            'username': current_user.username,
            'subscription_status': current_user.subscription_status,
            'upload_count': current_user.upload_count,
            'subscription_end_date': current_user.subscription_end_date
        }
        return render_template('account.html', 
                             user=user_data,
                             is_premium=current_user.subscription_status == 'active')
    except Exception as e:
        logger.error(f"Error in account route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Account Error"), 500

# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)