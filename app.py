import os
import re
import tempfile

import fitz  # PyMuPDF
import joblib
import nltk
import pytesseract
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS, cross_origin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdf2image import convert_from_path
from PIL import Image

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the models
xgb_model = joblib.load('pkl/xgb_model.pkl')
vectorizer = joblib.load('pkl/vectorizer.pkl')
label_encoder = joblib.load('pkl/label_encoder.pkl')

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf_file.save(temp_file.name)
    
    text = ""
    doc = fitz.open(temp_file.name)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    doc.close()
    os.remove(temp_file.name)
    
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Perform login validation here
        if username == "admin" and password == "admin":  # Example validation
            return redirect(url_for('upload'))
        else:
            return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        file = request.files['file']
        skill_set = request.form.get('skills', '').split(',')

        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(('png', 'jpg', 'jpeg')):
            text = extract_text_from_image(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = xgb_model.predict(vectorized_text)
        category = label_encoder.inverse_transform(prediction)[0]

        filtered = any(skill.lower() in processed_text for skill in skill_set)

        return jsonify({'category': category, 'filtered': filtered})

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
