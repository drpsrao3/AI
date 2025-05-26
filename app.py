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
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Setup logging with more detail
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
database_url = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.instance_path = os.path.join(os.path.dirname(__file__), 'instance')
os.makedirs(app.instance_path, exist_ok=True)

# Initialize SQLAlchemy, Migrate, and Bcrypt
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Razorpay configuration
try:
    razorpay_client = razorpay.Client(auth=(os.getenv('RAZORPAY_KEY_ID'), os.getenv('RAZORPAY_KEY_SECRET')))
    logger.debug("Razorpay client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Razorpay client: {str(e)}")
    raise

# Global model instances (load on-demand to save memory)
summarizer = None
nlp = None

# User model with subscription
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    razorpay_customer_id = db.Column(db.String(100))
    subscription_status = db.Column(db.String(20))
    subscription_id = db.Column(db.String(100))
    upload_count = db.Column(db.Integer, default=0)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
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
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                framework="pt",
                device=-1
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {str(e)}")
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
            logger.error(f"Failed to load NLP model: {str(e)}")
            raise
    return nlp

def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        start_time = time.time()
        with pdfplumber.open(pdf_path) as pdf:
            text = " ".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            logger.warning("pdfplumber extracted no text, falling back to PyPDF2")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
        if not text.strip():
            return "Error: PDF contains no extractable text"
        logger.info(f"Text extraction completed in {time.time() - start_time:.2f}s, length: {len(text)} characters")
        return text
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        logger.error(error_msg)
        return error_msg

def preprocess_text(text):
    try:
        text = ' '.join(text.split())
        noise_phrases = [
            "IN THE COURT OF", "CASE NO.", "JUDGMENT", 
            "BEFORE THE HON'BLE", "IN THE MATTER OF"
        ]
        for phrase in noise_phrases:
            text = text.replace(phrase, "")
        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return text

def summarize_legal_text(text):
    try:
        logger.info("Starting text summarization...")
        start_time = time.time()
        summarizer = load_summarizer()
        text = preprocess_text(text)[:100000]
        max_chunk = 1024
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = []
        chunk_limit = 10 if current_user.subscription_status == 'active' else 5
        for i, chunk in enumerate(chunks[:chunk_limit]):
            try:
                summary = summarizer(
                    chunk,
                    max_length=300,
                    min_length=150,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                summaries.append(summary)
                logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.warning(f"Error summarizing chunk {i+1}: {str(e)}")
                continue
        if not summaries:
            return "Error: Could not generate any summary chunks"
        full_summary = " ".join(summaries)
        logger.info(f"Summarization completed in {time.time() - start_time:.2f}s")
        return full_summary
    except Exception as e:
        error_msg = f"Error summarizing text: {str(e)}"
        logger.error(error_msg)
        return error_msg

def extract_legal_entities(text):
    try:
        logger.info("Extracting legal entities...")
        start_time = time.time()
        nlp = load_nlp()
        char_limit = 100000 if current_user.subscription_status == 'active' else 50000
        doc = nlp(text[:char_limit])
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
                if "J." in ent.text:
                    legal_tags["Judges"].add(ent.text)
                else:
                    legal_tags["Parties"].add(ent.text)
            elif ent.label_ == "ORG":
                if "court" in ent.text.lower():
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
            if any(c in sent.text for c in [" U.S. ", " F. ", " S.Ct. "]):
                legal_tags["Citations"].add(sent.text.strip())
        result = {k: sorted(v) for k, v in legal_tags.items() if v}
        logger.info(f"Entity extraction completed in {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        error_msg = f"Error extracting entities: {str(e)}"
        logger.error(error_msg)
        return {"Error": error_msg}

@app.route('/', methods=['GET'])
def index():
    try:
        logger.debug("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return "Internal Server Error", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            if username is None or password is None:
                flash('Username and password are required', 'error')
                return redirect(url_for('login'))
            username = username.strip()
            password = password.strip()
            logger.debug(f"Login form data: username={username}, password={'set' if password else 'None'}")
            if username != request.form.get('username') or password != request.form.get('password'):
                flash('Please remove any leading or trailing spaces in your username or password', 'error')
                return redirect(url_for('login'))
            if not username or not password:
                flash('Username and password are required', 'error')
                return redirect(url_for('login'))
            user = User.query.filter_by(username=username).first()
            logger.info(f"Login attempt for username: {username}, user found: {user is not None}")
            if user:
                logger.debug(f"Stored password hash: {user.password}")
                password_match = bcrypt.check_password_hash(user.password, password)
                logger.debug(f"Password match: {password_match}")
                if password_match:
                    login_user(user)
                    logger.info(f"User {username} logged in successfully")
                    flash('Logged in successfully', 'success')
                    return redirect(url_for('index'))
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        logger.debug("Rendering login page")
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login route: {str(e)}")
        return "Internal Server Error", 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            if username is None or password is None:
                flash('Username and password are required', 'error')
                return redirect(url_for('register'))
            username = username.strip()
            password = password.strip()
            logger.debug(f"Register attempt: username={username}, password={'set' if password else 'None'}")
            if username != request.form.get('username') or password != request.form.get('password'):
                flash('Please remove any leading or trailing spaces in your username or password', 'error')
                return redirect(url_for('register'))
            if not username or not password:
                flash('Username and password are required', 'error')
                return redirect(url_for('register'))
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return redirect(url_for('register'))
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, password=hashed_password, subscription_status='inactive', upload_count=0)
            try:
                db.session.add(user)
                db.session.commit()
                flash('Registration successful, please log in', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                flash(f'Registration failed: {str(e)}', 'error')
                logger.error(f"Registration error: {str(e)}")
                return redirect(url_for('register'))
        logger.debug("Rendering register page")
        return render_template('register.html')
    except Exception as e:
        logger.error(f"Error in register route: {str(e)}")
        return "Internal Server Error", 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        flash('Logged out successfully', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in logout route: {str(e)}")
        return "Internal Server Error", 500

@app.route('/create-subscription', methods=['POST'])
@login_required
def create_subscription():
    try:
        if not current_user.razorpay_customer_id:
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout))
            )
            def create_customer():
                return razorpay_client.customer.create({
                    'name': current_user.username,
                    'email': f'{current_user.username}@example.com',
                })
            
            customer = create_customer()
            current_user.razorpay_customer_id = customer['id']
            db.session.commit()
        else:
            customer = razorpay_client.customer.retrieve(current_user.razorpay_customer_id)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout))
        )
        def create_subscription_call():
            return razorpay_client.subscription.create({
                'plan_id': 'plan_ABC123XYZ',  # Replace with your actual Plan ID
                'customer_id': customer['id'],
                'total_count': 12,
                'quantity': 1,
                'notes': {'user_id': current_user.id}
            })

        subscription = create_subscription_call()
        logger.debug(f"Subscription created: {subscription['id']}")
        return render_template('payment.html', 
                              subscription_id=subscription['id'],
                              key_id=os.getenv('RAZORPAY_KEY_ID'))
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network error while creating subscription: {str(e)}")
        flash("Failed to connect to payment gateway. Please try again later.", 'error')
        return redirect(url_for('index'))
    except razorpay.errors.BadRequestError as e:
        logger.error(f"Razorpay API error: {str(e)}")
        flash(f"Payment error: {str(e)}", 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Unexpected error creating subscription: {str(e)}")
        flash(f"Error creating subscription: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/payment-success', methods=['POST'])
@login_required
def payment_success():
    try:
        subscription_id = request.form.get('razorpay_subscription_id')
        subscription = razorpay_client.subscription.fetch(subscription_id)
        if subscription['status'] == 'active':
            current_user.subscription_id = subscription_id
            current_user.subscription_status = 'active'
            db.session.commit()
            flash('Subscription successful!', 'success')
        else:
            flash('Subscription not activated', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in payment-success route: {str(e)}")
        flash(f"Payment error: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/webhook', methods=['POST'])
def razorpay_webhook():
    try:
        payload = request.get_json()
        event = payload['event']
        if event == 'subscription.activated':
            subscription = payload['payload']['subscription']['entity']
            customer_id = subscription['customer_id']
            user = User.query.filter_by(razorpay_customer_id=customer_id).first()
            if user:
                user.subscription_id = subscription['id']
                user.subscription_status = 'active'
                db.session.commit()
                logger.info(f"Subscription activated for user {user.username}")
        elif event == 'subscription.cancelled':
            subscription = payload['payload']['subscription']['entity']
            customer_id = subscription['customer_id']
            user = User.query.filter_by(razorpay_customer_id=customer_id).first()
            if user:
                user.subscription_status = 'inactive'
                db.session.commit()
                logger.info(f"Subscription cancelled for user {user.username}")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/manage-subscription')
@login_required
def manage_subscription():
    try:
        if current_user.subscription_id:
            subscription = razorpay_client.subscription.fetch(current_user.subscription_id)
            return render_template('manage_subscription.html', subscription=subscription)
        flash('No active subscription', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in manage-subscription route: {str(e)}")
        return "Internal Server Error", 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if current_user.subscription_status != 'active' and current_user.upload_count >= 5:
            flash('Free plan limit reached. Subscribe to Premium for unlimited uploads.', 'error')
            return redirect(url_for('index'))
        
        if 'file' not in request.files:
            flash('No file part in the request', 'error')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        if not file or not allowed_file(file.filename):
            flash('Only PDF files are allowed', 'error')
            return redirect(url_for('index'))
        filename = secure_filename(file.filename)
        unique_id = str(int(time.time()))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        try:
            file.save(filepath)
            logger.info(f"File saved temporarily at: {filepath}")
        except Exception as e:
            flash('Failed to save file', 'error')
            logger.error(f"File save error: {str(e)}")
            return redirect(url_for('index'))
        try:
            text = extract_text_from_pdf(filepath)
            if text.startswith("Error"):
                flash(text, 'error')
                return redirect(url_for('index'))
            summary = summarize_legal_text(text)
            if summary.startswith("Error"):
                flash(summary, 'error')
                return redirect(url_for('index'))
            tags = extract_legal_entities(text)
            if current_user.subscription_status != 'active':
                current_user.upload_count += 1
                db.session.commit()
            return render_template(
                'result.html',
                summary=summary,
                tags=tags,
                original_filename=filename,
                text_preview=text[:500] + "..." if len(text) > 500 else text
            )
        except Exception as e:
            flash(f"Processing error: {str(e)}", 'error')
            logger.error(f"Processing error: {str(e)}")
            return redirect(url_for('index'))
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Temporary file removed: {filepath}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        return "Internal Server Error", 500

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)