from flask import Flask, request, render_template
import PyPDF2
import os
import spacy
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Lazy loading: models initialized only when first needed
summarizer = None
nlp = None

def load_summarizer():
    global summarizer
    if summarizer is None:
        # Use a lighter model to avoid OOM
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
    return summarizer

def load_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def summarize_text(text):
    try:
        summarizer = load_summarizer()
        max_chunk = 1000
        text = text[:4000]
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        return " ".join(summaries)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def tag_entities(text):
    try:
        nlp = load_nlp()
        doc = nlp(text[:10000])
        tags = {"Parties": [], "Dates": [], "Laws": []}
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                tags["Parties"].append(ent.text)
            elif ent.label_ == "DATE":
                tags["Dates"].append(ent.text)
            elif ent.label_ == "LAW":
                tags["Laws"].append(ent.text)
        return tags
    except Exception as e:
        return {"Error": f"Error tagging entities: {str(e)}"}

@app.route('/')
def home():
    return "AI Law Service Running"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('result.html', summary="No file uploaded", tags={})
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', summary="No file selected", tags={})
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        if text.startswith("Error"):
            return render_template('result.html', summary=text, tags={})
        summary = summarize_text(text)
        if summary.startswith("Error"):
            return render_template('result.html', summary=summary, tags={})
        tags = tag_entities(text)
        return render_template('result.html', summary=summary, tags=tags)
    return render_template('result.html', summary="Invalid file format (PDF required)", tags={})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
