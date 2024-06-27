---

# Resume Classification with XGBoost

This project demonstrates a machine learning approach to classify resumes into different categories using XGBoost. Resumes are preprocessed and vectorized to train an XGBoost model, which is evaluated for accuracy in predicting job categories.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

The project focuses on automating the categorization of resumes based on their content. It preprocesses textual data, extracts features using TF-IDF vectorization, addresses class imbalance with SMOTE, and trains an XGBoost classifier to predict job categories accurately.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zayn2003/Resume-classification.git
   cd Resume-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```bash
   python -m nltk.downloader stopwords wordnet
   ```

## Usage

1. Place your resume dataset (`Resume.csv`) in the project directory.
2. Run the preprocessing and model training script
3. Evaluate the trained model and view classification results.

## Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- nltk

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
