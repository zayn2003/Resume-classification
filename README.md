# BOL-Handling
### Project Description for GitHub Repository

---

# Automated Processing and Verification of Export Documents

This project aims to automate the processing and verification of export-related documents such as invoices and bills of lading using machine learning, OCR, and NLP techniques. The primary goal is to reduce the time and errors associated with manual document handling.

## Features

- **Data Collection and Preprocessing**: Collects and preprocesses export-related documents from diverse sources.
- **OCR Integration**: Uses Optical Character Recognition (OCR) to convert scanned images of documents into machine-readable text.
- **NLP Techniques**: Applies Natural Language Processing (NLP) to extract key information from the text, such as shipper, consignee, and shipment details.
- **Logistic Regression Model**: Trains a logistic regression model to classify entries as correct or incorrect, identifying errors and anomalies.
- **Real-time Processing**: Deploys the trained model using Flask for real-time document processing and verification.
- **Continuous Improvement**: Implements a feedback loop to continuously update and improve the models based on user feedback.

## Datasets

The project uses datasets containing export-related documents, including information on shipments, suppliers, and companies.

## How to Use

1. **Clone the Repository**: `git clone https://github.com/zayn2003/BOL-Handling.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run the Application**: `python app.py`
4. **Upload Documents**: Use the provided interface to upload and process documents.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Tesseract OCR
- Natural Language Processing (NLP)
- Logistic Regression
- Flask

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---
