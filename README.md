🚀 HexSoftwares_TechSupportHubBot
📌 Project Overview

TechSupport Hub Bot is an AI-powered customer support chatbot developed as part of the internship assignment for Hex Softwares Pvt. Ltd.

The chatbot uses Natural Language Processing (NLP) and Machine Learning techniques to classify user queries into predefined support intents and provide appropriate responses through a Flask-based web interface.

🎯 Project Objective

To develop a virtual AI agent capable of:

Handling customer inquiries

Managing account access issues

Resolving billing and subscription queries

Providing technical support

Guiding users with feature usage

Managing privacy and data export requests

Handling greetings, feedback, and out-of-scope queries

🧠 Technologies Used

As defined in requirements.txt 

requirements

Python 3.10+

Scikit-learn

NumPy

Flask

Flask-CORS

🏗️ System Architecture

The chatbot consists of:

1️⃣ NLP Processing

Implemented in:

smartassist_chatbot.py 

smartassist_chatbot

Includes:

Text cleaning

Tokenization

Stopword removal

Lemmatization

2️⃣ Intent Classification

TF-IDF Vectorization

Logistic Regression / Ensemble Classifier

Confidence threshold handling

Cross-validation support

3️⃣ Training Dataset

Defined in:

techsupport_training_data.py 

techsupport_training_data

Includes:

600+ labeled examples

12 different support intents

Multiple variations per intent

4️⃣ Web Interface

Implemented in:

app.py 

app

Provides:

Interactive chat UI

Real-time message processing

Response display

5️⃣ Accuracy Testing

Implemented in:

test_improved_bot.py 

test_improved_bot

Includes:

59 diverse test cases

Accuracy scoring

Performance grading

Confidence analysis

📊 Model Performance

Cross-Validation Accuracy: ~68–70%

Test Accuracy: ~93%

12 Supported Intents

Balanced Multi-Class Classification

📁 Project Structure


HexSoftwares_TechSupportHubBot/
│
├── app.py
├── smartassist_chatbot.py
├── techsupport_training_data.py
├── test_improved_bot.py
├── requirements.txt
├── README.md
│
├── templates/
└── static/

🌐 How To Run The Project
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/HexSoftwares_TechSupportHubBot.git
cd HexSoftwares_TechSupportHubBot

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Web Application
python app.py


Open in browser:

http://localhost:5000

🧪 Run Accuracy Test
python test_improved_bot.py
