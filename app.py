import os
import tempfile
from flask import Flask, request, jsonify
from pdf2image import convert_from_path
import pytesseract
import joblib
from PIL import Image
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from flask_cors import CORS, cross_origin
import fitz  # PyMuPDF

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the models
xgb_model = joblib.load('/pkl/xgb_model.pkl')
vectorizer = joblib.load('/pkl/vectorizer.pkl')
label_encoder = joblib.load('/pkl/label_encoder.pkl')

def extract_text_from_pdf(pdf_file):
    # Save the FileStorage object to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf_file.save(temp_file.name)
    
    text = ""
    doc = fitz.open(temp_file.name)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    doc.close()  # Close the PDF document
    os.remove(temp_file.name)  # Delete the temporary file
    
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()  # Tokenize text into words
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return ' '.join(tokens)  # Return processed text as a single string

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    file = request.files['file']
    skill_set = request.form.get('skills', '').split(',')

    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(('png', 'jpg', 'jpeg')):
        text = extract_text_from_image(file)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Process the extracted text with the model (add your text processing and classification code here)
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = xgb_model.predict(vectorized_text)
    category = label_encoder.inverse_transform(prediction)[0]

    # Filter by skills (add your skill filtering code here)
    filtered = any(skill.lower() in processed_text for skill in skill_set)

    return jsonify({'category': category, 'filtered': filtered})

if __name__ == '__main__':
    app.run(debug=True)
