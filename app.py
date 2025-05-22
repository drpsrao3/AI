from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
import spacy
import pdfplumber
from transformers import pipeline
from werkzeug.utils import secure_filename
import logging
import time

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Secret key for sessions

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')  # SQLite locally, PostgreSQL on Render
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and Bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instances
summarizer = None
nlp = None

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # Stores hashed password

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Loading summarization model...")
        try:
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                framework="pt",
                device=-1  # Use CPU
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
            return "Error: PDF contains no extractable text"
        logger.info(f"Text extraction completed in {time.time() - start_time:.2f}s, length: {len(text)} characters")
        return text
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        logger.error(error_msg)
        return error_msg

def preprocess_text(text):
    """Clean and prepare legal text for summarization"""
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
        for i, chunk in enumerate(chunks[:10]):
            try:
                summary = summarizer(
                    chunk,
                    max_length=300,  # Increased for more complete summaries
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
        doc = nlp(text[:100000])  # Increased for more complete entity extraction
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
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful, please log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
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
        flash(f"Unexpected error: {str(e)}", 'error')
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        return redirect(url_for('index'))

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)