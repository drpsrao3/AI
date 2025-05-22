from flask import Flask, request, render_template, redirect, url_for, flash
import os
import spacy
import pdfplumber  # Added for PDF text extraction
from transformers import pipeline
from werkzeug.utils import secure_filename
import logging
import time

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Secret key for flash messages

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instances
summarizer = None
nlp = None

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
                device=-1  # Use CPU (change to 0 for GPU if available)
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
        # Basic cleaning
        text = ' '.join(text.split())
        
        # Remove common legal document noise
        noise_phrases = [
            "IN THE COURT OF", "CASE NO.", "JUDGMENT", 
            "BEFORE THE HON'BLE", "IN THE MATTER OF"
        ]
        for phrase in noise_phrases:
            text = text.replace(phrase, "")
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return text  # Return original if preprocessing fails

def summarize_legal_text(text):
    try:
        logger.info("Starting text summarization...")
        start_time = time.time()
        summarizer = load_summarizer()
        
        # Preprocess and truncate to reasonable length
        text = preprocess_text(text)[:100000]  # Limit to first 100k characters
        
        # Chunking parameters
        max_chunk = 1024
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        
        summaries = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to 10 chunks max
            try:
                summary = summarizer(
                    chunk,
                    max_length=200,
                    min_length=100,
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
        
        # Process first 50k characters for efficiency
        doc = nlp(text[:50000])
        
        legal_tags = {
            "Parties": set(),
            "Judges": set(),
            "Courts": set(),
            "Dates": set(),
            "Laws": set(),
            "Case Numbers": set(),
            "Citations": set()
        }
        
        # Entity extraction
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if "J." in ent.text:  # Judge name pattern
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
        
        # Additional pattern matching
        for sent in doc.sents:
            # Case name pattern (Plaintiff v. Defendant)
            if " v. " in sent.text:
                parts = [p.strip() for p in sent.text.split(" v. ")]
                if len(parts) == 2:
                    legal_tags["Parties"].update(parts)
            
            # Citation pattern (e.g., 123 U.S. 456)
            if any(c in sent.text for c in [" U.S. ", " F. ", " S.Ct. "]):
                legal_tags["Citations"].add(sent.text.strip())
        
        # Convert sets to sorted lists
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

@app.route('/upload', methods=['POST'])
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
        
        # Secure filename and create unique filepath
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
        
        # Process the file
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)