# AI-Medical-Chatbot
An AI-Based Medical Chatbot Model for Infectious Disease Prediction
**AI Infectious Disease Chatbot**

This project is an AI-based chatbot designed to assist users in predicting potential infectious diseases based on their input symptoms. The chatbot utilizes machine learning algorithms and natural language processing (NLP) techniques to analyze symptoms and assess the likelihood of an infectious disease. Additionally, it provides precautionary advice based on the predicted disease to help users take necessary preventive measures.

**Project Structure**
The chatbot relies on various datasets containing information about infectious diseases, symptom descriptions, and precautionary measures. The system is structured to:

- Load and preprocess training and testing data.
- Use deep learning models for disease prediction.
- Provide text-based responses in a conversational format.
- Offer multilingual support using the Google Translate API.
- Display disease descriptions and precautionary advice based on the prediction.

**Files in the Project**

- `covid-19.json`: Contains a dataset of symptoms and responses related to COVID-19 and other infectious diseases.
- `model/model.h5`: Pre-trained deep learning model for chatbot responses.
- `Dataset/symptom_description.csv`: Provides descriptions of each symptom.
- `Dataset/symptom_precaution.csv`: Lists precautionary measures for various infectious diseases.

**Libraries Used**

- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For feature extraction and data processing.
- `TensorFlow/Keras`: For building and training the deep learning model.
- `speech_recognition`: For voice input functionality.
- `googletrans`: For multilingual support.
- `matplotlib`: For visualizing training results.

**Code Example**

```python
import os
import json
import numpy as np
import pandas as pd
import pickle
import re
import speech_recognition as sr
from keras import layers, models, preprocessing, optimizers
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator

# Load dataset
translator = Translator()
recognizer = sr.Recognizer()
with open("Dataset/covid-19.json") as f:
    json_data = json.load(f)["intents"]

X, Y = [], []
for entry in json_data:
    for question in entry['patterns']:
        for response in entry['responses']:
            X.append(question.lower().strip())
            Y.append(response)

# Tokenization and model training
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None)
tfidf = vectorizer.fit_transform(X).toarray()
```

**How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Infectious-Disease-Chatbot.git
   ```
2. Install required libraries:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run the chatbot script:
   ```bash
   python manage.py runserver
   ```

**How It Works**

- The chatbot prompts users to input symptoms.
- It predicts potential infectious diseases using an AI model.
- The chatbot provides information about the disease and necessary precautions.
- Users can interact via text or voice input.
- The chatbot supports multiple languages using Google Translate.

**Precautionary Advice**
The chatbot suggests precautionary measures based on the identified infectious disease, helping users take appropriate action to prevent further spread.

**Author**

- **Sri Ram Reddy Alla**: Developer of the AI Infectious Disease Chatbot.

**About**
This project is designed to assist in early disease detection and prevention. Future enhancements include integrating more datasets, improving NLP capabilities, and adding real-time health monitoring features.

**Topics**

- artificial-intelligence
- deep-learning
- NLP
- disease-prediction
- chatbot

**License**
MIT License

**GNU general public license 3.0**

The GNU General Public License v3.0 (GPL-3.0) is a free software license that ensures software freedom and protects user rights. It allows users to freely run, modify, and distribute the software, provided that any derivative work is also licensed under GPL-3.0. This ensures that the software remains open-source and prevents proprietary restrictions.
