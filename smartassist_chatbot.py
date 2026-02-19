"""
SmartAssist AI - Intelligent Customer Support Chatbot
Features:
- NLP Preprocessing
- TF-IDF & DistilBERT Intent Classification
- Sentiment Analysis (TextBlob, VADER, and trained models)
- Confidence Score Threshold
- Smart Response Mapping
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
import torch
import warnings

# ========== NEW IMPORTS FOR ENHANCED CLASSIFIER ==========
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib
from collections import defaultdict
import random
# ========================================================

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class NLPPreprocessor:
    """Handles text preprocessing for the chatbot"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Common contractions
        self.contractions = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "didn't": "did not", "doesn't": "does not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would",
            "you'd": "you would", "he'd": "he would", "she'd": "she would",
            "we'd": "we would", "they'd": "they would"
        }
    
    def expand_contractions(self, text):
        """Expand contractions"""
        text = text.lower()
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
        except:
            nltk.download('punkt')
            tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, remove_stops=True):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords (optional)
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Join back to string
        return ' '.join(tokens)


class IntentClassifier:
    """Enhanced intent classification with multiple models and ensembling"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.preprocessor = NLPPreprocessor()
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        self.intent_labels = []
        self.cv_scores = []
    
    def _get_optimal_vectorizer_params(self, texts):
        """Automatically determine optimal vectorizer parameters based on dataset size"""
        n_samples = len(texts)
        
        if n_samples < 20:
            # Very small dataset - most lenient
            return {
                'max_features': 500,
                'ngram_range': (1, 1),
                'min_df': 1,
                'max_df': 1.0,
                'use_idf': False,
                'smooth_idf': False,
                'stop_words': None,
                'sublinear_tf': False
            }
        elif n_samples < 100:
            # Small dataset - lenient
            return {
                'max_features': 2000,
                'ngram_range': (1, 2),
                'min_df': 1,
                'max_df': 0.95,
                'use_idf': True,
                'smooth_idf': True,
                'stop_words': 'english',
                'sublinear_tf': True
            }
        else:
            # Adequate dataset - normal parameters
            return {
                'max_features': 8000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.8,
                'use_idf': True,
                'smooth_idf': True,
                'stop_words': 'english',
                'sublinear_tf': True
            }
    
    def train(self, training_data, use_ensemble=True, optimize=True):
        """
        Train with advanced techniques
        training_data: list of tuples [(text, intent), ...]
        """
        # Preprocess all texts
        texts = [self.preprocessor.preprocess(text) for text, _ in training_data]
        intents = [intent for _, intent in training_data]
        
        print(f"Training on {len(texts)} examples with {len(set(intents))} intents")
        
        # Get optimal vectorizer parameters based on dataset size
        params = self._get_optimal_vectorizer_params(texts)
        self.vectorizer = TfidfVectorizer(**params)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(intents)
        self.intent_labels = self.label_encoder.classes_
        
        # Convert texts to vectors with error handling
        try:
            X = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            if "After pruning, no terms remain" in str(e):
                print("WARNING: No terms remain after pruning. Using fallback parameters...")
                # Ultimate fallback
                self.vectorizer = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0,
                    use_idf=False,
                    stop_words=None,
                    lowercase=True,
                    token_pattern=r'(?u)\b\w+\b'
                )
                X = self.vectorizer.fit_transform(texts)
            else:
                raise e
        
        # Check if we have any features
        if X.shape[1] == 0:
            raise ValueError("No features extracted from training data! Check your preprocessing.")
        
        print(f"Extracted {X.shape[1]} features from {len(texts)} documents")
        
        # Setup classifier based on dataset size
        n_classes = len(set(y_encoded))
        n_samples = len(y_encoded)
        
        if use_ensemble and n_samples >= 20:
            # Create ensemble of multiple classifiers
            clf1 = LogisticRegression(max_iter=3000, C=1.5, class_weight='balanced')
            clf2 = MultinomialNB(alpha=0.1)
            clf3 = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                         random_state=42, class_weight='balanced')
            clf4 = SVC(kernel='linear', probability=True, class_weight='balanced')
            
            self.classifier = VotingClassifier(
                estimators=[
                    ('lr', clf1),
                    ('nb', clf2),
                    ('rf', clf3),
                    ('svm', clf4)
                ],
                voting='soft',
                weights=[2, 1, 2, 2]
            )
        else:
            # Single classifier for small datasets
            if optimize and n_samples >= 30:
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'sag'],
                    'max_iter': [2000],
                    'class_weight': ['balanced', None]
                }
                base_clf = LogisticRegression()
                self.classifier = GridSearchCV(
                    base_clf, param_grid, cv=min(5, n_classes), 
                    scoring='accuracy', n_jobs=-1
                )
            else:
                self.classifier = LogisticRegression(
                    C=1.0, 
                    max_iter=2000,
                    class_weight='balanced',
                    multi_class='multinomial',
                    solver='lbfgs'
                )
        
        # Cross-validation (if enough data)
        if n_samples >= 10:
            cv_folds = min(5, n_samples // 2) if n_samples >= 10 else 2
            try:
                cv_scores = cross_val_score(self.classifier, X, y_encoded, cv=cv_folds)
                self.cv_scores = cv_scores
                print(f"Cross-validation scores: {cv_scores}")
                print(f"Mean CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
            except Exception as e:
                print(f"Could not perform cross-validation: {e}")
                self.cv_scores = []
        
        # Fit on full data
        self.classifier.fit(X, y_encoded)
        
        # Create pipeline for easy prediction
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
    
    def predict(self, text):
        """Predict intent with confidence calibration"""
        if self.pipeline is None:
            return "unknown", 0.0
        
        # Preprocess the input text
        processed_text = self.preprocessor.preprocess(text)
        
        try:
            # Get probabilities from all classifiers
            proba = self.pipeline.predict_proba([processed_text])[0]
            
            # Get the highest probability
            max_proba = np.max(proba)
            
            # Get the intent with highest probability
            intent_idx = np.argmax(proba)
            intent = self.intent_labels[intent_idx]
            confidence = float(max_proba)
            
            # Check if we have enough confidence
            if confidence < self.confidence_threshold:
                return "unknown", confidence
            
            # Check for ambiguity (two intents too close)
            sorted_proba = np.sort(proba)[::-1]
            if len(sorted_proba) > 1:
                if sorted_proba[0] - sorted_proba[1] < 0.15:
                    return "ambiguous", confidence
            
            return intent, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown", 0.0
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'label_encoder': self.label_encoder,
                'intent_labels': self.intent_labels,
                'confidence_threshold': self.confidence_threshold,
                'vectorizer': self.vectorizer,
                'cv_scores': self.cv_scores
            }, f)
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pipeline = data['pipeline']
            self.label_encoder = data['label_encoder']
            self.intent_labels = data['intent_labels']
            self.confidence_threshold = data['confidence_threshold']
            self.vectorizer = data['vectorizer']
            self.cv_scores = data.get('cv_scores', [])


class DistilBERTIntentClassifier:
    """Advanced intent classification using DistilBERT"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.intent_labels = []
        
    def initialize_pretrained(self):
        """Initialize with pretrained sentiment model"""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.classifier = pipeline("sentiment-analysis", 
                                     model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
            print("Falling back to rule-based sentiment")
            self.classifier = None
        
    def predict(self, text):
        """Predict intent using DistilBERT"""
        if self.classifier is None:
            return "unknown", 0.0
        
        try:
            result = self.classifier(text[:512])[0]
            confidence = result['score']
            label = result['label']
            
            if confidence < self.confidence_threshold:
                return "unknown", confidence
                
            return label.lower(), confidence
        except Exception as e:
            print(f"DistilBERT prediction error: {e}")
            return "unknown", 0.0


class SentimentAnalyzer:
    """Multi-method sentiment analysis"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.transformer_pipeline = None
        
    def analyze_textblob(self, text):
        """Sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            'method': 'TextBlob',
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_vader(self, text):
        """Sentiment analysis using VADER"""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            'method': 'VADER',
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_transformers(self, text):
        """Sentiment analysis using DistilBERT transformer"""
        try:
            if self.transformer_pipeline is None:
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            
            result = self.transformer_pipeline(text[:512])[0]
            
            return {
                'method': 'DistilBERT',
                'sentiment': result['label'].lower(),
                'confidence': result['score']
            }
        except Exception as e:
            return {
                'method': 'DistilBERT',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_all(self, text):
        """Run all sentiment analysis methods"""
        return {
            'textblob': self.analyze_textblob(text),
            'vader': self.analyze_vader(text),
            'transformer': self.analyze_transformers(text)
        }
    
    def get_consensus_sentiment(self, text):
        """Get consensus sentiment from all methods"""
        results = self.analyze_all(text)
        
        sentiments = [
            results['textblob']['sentiment'],
            results['vader']['sentiment'],
            results['transformer']['sentiment']
        ]
        
        # Return most common sentiment
        from collections import Counter
        consensus = Counter(sentiments).most_common(1)[0][0]
        
        return {
            'consensus': consensus,
            'individual_results': results
        }


class ResponseMapper:
    """Smart response mapping based on intent and sentiment"""
    
    def __init__(self):
        # Define responses for different intents
        self.intent_responses = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Welcome! How may I help you?",
                "Good to see you! How can I assist?",
                "Hello! I'm here to help. What do you need?"
            ],
            'farewell': [
                "Goodbye! Have a great day!",
                "Thank you for chatting. Take care!",
                "See you later! Feel free to reach out anytime.",
                "Bye! Hope I was helpful!",
                "Have a wonderful day! Goodbye!"
            ],
            'product_inquiry': [
                "I'd be happy to help with product information. What would you like to know?",
                "Sure! Which product are you interested in?",
                "I can help you with that. Please tell me more about what you're looking for.",
                "We have several great products. What features interest you?",
                "I'd love to tell you about our products. What's your specific question?"
            ],
            'complaint': [
                "I'm sorry to hear you're experiencing issues. Let me help resolve this.",
                "I understand your concern. I'll do my best to assist you.",
                "Thank you for bringing this to our attention. How can I help fix this?",
                "I apologize for the inconvenience. Let me make this right.",
                "I hear your frustration and I'm here to help solve this problem."
            ],
            'billing': [
                "I can help with billing questions. What do you need to know?",
                "Let me assist you with your billing inquiry.",
                "I'm here to help with billing matters. What's your question?",
                "I can check your billing details. Could you provide more information?",
                "I'll help you resolve this billing issue. Please tell me more."
            ],
            'technical_support': [
                "I'll help you troubleshoot this issue. Can you describe the problem?",
                "Let me assist with your technical issue. What seems to be the problem?",
                "I'm here to help with technical support. What's happening?",
                "I can help fix that. When did this issue start?",
                "Let's troubleshoot together. What error message are you seeing?"
            ],
            'order_tracking': [
                "I'll help you track your order. Can you provide your order number?",
                "Let me check on your shipment. What's your order number?",
                "I'd be happy to check your order status. Do you have your order ID?",
                "I can track that for you. Please share your order number.",
                "Let me look up your order. Could you give me your order number?"
            ],
            'account_issue': [
                "I can help with account issues. What specific problem are you experiencing?",
                "Let me assist you with your account. Can you describe the issue?",
                "I understand account problems can be frustrating. I'll help you fix this.",
                "I can help with login and account access. What's happening?",
                "Let me check your account status. What seems to be the problem?"
            ],
            'pricing': [
                "I can help with pricing information. Which product are you interested in?",
                "Our pricing varies by product. What specific item are you looking at?",
                "I'd be happy to explain our pricing. What would you like to know?",
                "Let me get you accurate pricing. Which product interests you?",
                "I can provide detailed pricing information. Tell me what you need."
            ],
            'shipping': [
                "I can help with shipping questions. Where are you located?",
                "Shipping times depend on your location. Where should I ship to?",
                "I'll check shipping options for you. What's your delivery address?",
                "We offer several shipping methods. What's your preferred speed?",
                "Let me get shipping details for your location."
            ],
            'returns_refund': [
                "I can help with returns and refunds. What item do you want to return?",
                "I understand you want to return something. Let me guide you through the process.",
                "I'll help you process this return. When did you receive the item?",
                "Returns are easy. Let me explain the process.",
                "I can initiate a refund for you. Please provide your order number."
            ],
            'positive_feedback': [
                "Thank you! I'm glad to hear that! 😊",
                "That's wonderful to hear! Thank you for your feedback!",
                "We appreciate your kind words! Is there anything else I can help with?",
                "Thank you! We work hard to provide great service.",
                "I'll make sure to share your feedback with the team!"
            ],
            'negative_feedback': [
                "I'm sorry to hear that. Thank you for your honest feedback.",
                "I appreciate you telling us about this. How can I make it right?",
                "Thank you for bringing this to our attention. Let me help resolve your concerns.",
                "I understand your frustration. Let me help fix this for you.",
                "Your feedback helps us improve. I'm here to help with your specific issue."
            ],
            'small_talk': [
                "I'm SmartAssist AI, your virtual assistant!",
                "I can help with product information, orders, billing, and technical support.",
                "I'm here to answer questions and help solve problems.",
                "I'm a chatbot designed to assist you with customer support.",
                "You can ask me about our products, your orders, billing issues, and more!"
            ],
            'unknown': [
                "I'm not sure I understood that. Could you rephrase?",
                "I didn't quite catch that. Can you tell me more?",
                "Could you provide more details so I can better assist you?",
                "I'm not sure how to help with that. Can you try asking differently?",
                "I don't have an answer for that yet. Could you ask in another way?"
            ],
            'ambiguous': [
                "I'm not entirely sure what you need. Could you be more specific?",
                "I see a few possible interpretations. Can you clarify?",
                "Just to make sure I understand correctly, could you rephrase that?",
                "I want to help, but I need a bit more clarity. What exactly do you need?",
                "I'm seeing multiple possible intents. Could you be more specific?"
            ]
        }
        
        # Sentiment-adjusted response modifiers
        self.sentiment_modifiers = {
            'positive': {
                'prefix': "Great to hear! ",
                'suffix': " 😊"
            },
            'negative': {
                'prefix': "I understand your frustration. ",
                'suffix': " I'm here to help resolve this."
            },
            'neutral': {
                'prefix': "",
                'suffix': ""
            }
        }
    
    def get_response(self, intent, sentiment='neutral', confidence=1.0):
        """Get appropriate response based on intent and sentiment"""
        # Handle special cases
        if intent == 'unknown' and confidence < 0.3:
            return "I'm not sure I understand. Could you please rephrase your question?"
        
        if intent == 'ambiguous':
            return "I'm seeing multiple possible meanings. Could you be more specific?"
        
        # Get base response
        if intent in self.intent_responses:
            base_response = np.random.choice(self.intent_responses[intent])
        else:
            base_response = np.random.choice(self.intent_responses['unknown'])
        
        # Apply sentiment modifiers for certain intents
        if intent in ['complaint', 'technical_support', 'account_issue', 'negative_feedback'] and sentiment == 'negative':
            modifier = self.sentiment_modifiers['negative']
            response = modifier['prefix'] + base_response + modifier['suffix']
        elif sentiment == 'positive' and intent in ['greeting', 'product_inquiry', 'positive_feedback']:
            modifier = self.sentiment_modifiers['positive']
            response = modifier['prefix'] + base_response + modifier['suffix']
        else:
            response = base_response
        
        # Add confidence indicator if low
        if confidence < 0.6:
            response += " (If I misunderstood, please let me know!)"
        
        return response
    
    def add_intent_response(self, intent, responses):
        """Add custom intent responses"""
        self.intent_responses[intent] = responses


class SmartAssistAI:
    """Main chatbot class integrating all components"""
    
    def __init__(self, use_distilbert=False, confidence_threshold=0.7):
        self.preprocessor = NLPPreprocessor()
        self.use_distilbert = use_distilbert
        
        if use_distilbert:
            self.intent_classifier = DistilBERTIntentClassifier(confidence_threshold)
            self.intent_classifier.initialize_pretrained()
        else:
            # Use the ENHANCED classifier
            self.intent_classifier = IntentClassifier(confidence_threshold)
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_mapper = ResponseMapper()
        self.conversation_history = []
        self.training_stats = {}
    
    def train_intent_classifier(self, training_data, use_ensemble=True, optimize=True):
        """
        Train the intent classifier with enhanced options
        
        Args:
            training_data: List of (text, intent) tuples
            use_ensemble: Whether to use ensemble of classifiers (default: True)
            optimize: Whether to perform hyperparameter optimization (default: True)
        """
        if not self.use_distilbert:
            print("=" * 60)
            print("TRAINING INTENT CLASSIFIER")
            print("=" * 60)
            print(f"Training samples: {len(training_data)}")
            
            # Count intents
            intent_counts = {}
            for _, intent in training_data:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            print(f"Intents: {len(intent_counts)}")
            for intent, count in sorted(intent_counts.items())[:10]:
                print(f"  {intent}: {count} examples")
            if len(intent_counts) > 10:
                print(f"  ... and {len(intent_counts) - 10} more intents")
            
            print("-" * 60)
            
            # Train the classifier
            self.intent_classifier.train(
                training_data, 
                use_ensemble=use_ensemble, 
                optimize=optimize
            )
            
            # Store training stats
            if hasattr(self.intent_classifier, 'cv_scores') and self.intent_classifier.cv_scores is not None:
                cv_scores = self.intent_classifier.cv_scores
                if len(cv_scores) > 0:
                    self.training_stats = {
                        'cv_scores': cv_scores,
                        'mean_accuracy': cv_scores.mean(),
                        'std_accuracy': cv_scores.std(),
                        'training_size': len(training_data),
                        'num_intents': len(intent_counts)
                    }
                    print(f"✅ Training complete! Cross-validation accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
            else:
                print("✅ Training complete!")
            
            print("=" * 60 + "\n")
        else:
            print("DistilBERT model is pretrained. Fine-tuning not implemented in this version.")
    
    def process_message(self, user_message):
        """Process user message and generate response"""
        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'message': user_message
        })
        
        # Detect intent
        intent, confidence = self.intent_classifier.predict(user_message)
        
        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.get_consensus_sentiment(user_message)
        sentiment = sentiment_results['consensus']
        
        # Generate response
        response = self.response_mapper.get_response(intent, sentiment, confidence)
        
        # Store bot response
        self.conversation_history.append({
            'role': 'assistant',
            'message': response
        })
        
        # Return detailed analysis
        return {
            'user_message': user_message,
            'intent': intent,
            'intent_confidence': confidence,
            'sentiment': sentiment,
            'sentiment_details': sentiment_results,
            'response': response
        }
    
    def chat(self, message):
        """Simple chat interface"""
        result = self.process_message(message)
        return result['response']
    
    def get_detailed_analysis(self, message):
        """Get detailed analysis of message"""
        return self.process_message(message)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self):
        """Get conversation history"""
        return self.conversation_history
    
    def save_models(self, filepath_prefix):
        """Save trained models"""
        if not self.use_distilbert:
            self.intent_classifier.save_model(f"{filepath_prefix}_intent_classifier.pkl")
    
    def get_training_stats(self):
        """Get training statistics"""
        return self.training_stats
    
    def analyze_errors(self, test_data):
        """Analyze misclassifications to improve model"""
        errors = []
        confusion = {}
        
        for text, true_intent in test_data[:100]:  # Limit to 100 examples
            pred_intent, confidence = self.intent_classifier.predict(text)
            
            if pred_intent != true_intent:
                errors.append({
                    'text': text,
                    'true': true_intent,
                    'predicted': pred_intent,
                    'confidence': confidence
                })
                
                key = f"{true_intent} → {pred_intent}"
                confusion[key] = confusion.get(key, 0) + 1
        
        print("\n📊 ERROR ANALYSIS:")
        print(f"Total errors: {len(errors)}")
        
        if errors:
            print("\nTop confusion pairs:")
            sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
            for pair, count in sorted_confusion[:5]:
                print(f"  {pair}: {count} errors")
            
            print("\nExample errors:")
            for error in errors[:3]:
                print(f"  Text: '{error['text']}'")
                print(f"    True: {error['true']}, Predicted: {error['predicted']} ({error['confidence']:.2%})")
        
        return errors


# Data augmentation function
def augment_training_data(training_data, target_per_intent=30):
    """
    Automatically expand training data to reach target examples per intent
    """
    # Group by intent
    intent_examples = defaultdict(list)
    for text, intent in training_data:
        intent_examples[intent].append(text)
    
    expanded_data = []
    
    for intent, examples in intent_examples.items():
        # Add original examples
        for text in examples:
            expanded_data.append((text, intent))
        
        current_count = len(examples)
        
        # Generate variations if we need more
        if current_count < target_per_intent:
            variations_needed = target_per_intent - current_count
            
            for _ in range(variations_needed):
                # Pick a random example to vary
                original = random.choice(examples)
                
                # Create variation
                words = original.split()
                
                # Simple variations
                if len(words) > 2 and random.random() > 0.5:
                    # Swap two words
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
                    variation = ' '.join(words)
                elif random.random() > 0.5:
                    # Add filler word
                    fillers = ['actually', 'basically', 'just', 'like', 'so', 'really']
                    pos = random.randint(0, len(words))
                    words.insert(pos, random.choice(fillers))
                    variation = ' '.join(words)
                else:
                    # Add please at the end
                    variation = original + ' please'
                
                expanded_data.append((variation, intent))
    
    return expanded_data


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("SmartAssist AI - Enhanced Version")
    print("=" * 60)
    
    # Sample training data
    training_data = [
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("hey", "greeting"),
        ("good morning", "greeting"),
        ("goodbye", "farewell"),
        ("bye", "farewell"),
        ("see you", "farewell"),
        ("help", "technical_support"),
        ("not working", "technical_support"),
        ("issue", "technical_support"),
        ("refund", "billing"),
        ("charge", "billing"),
        ("payment", "billing"),
    ]
    
    # Initialize chatbot
    chatbot = SmartAssistAI(use_distilbert=False, confidence_threshold=0.7)
    
    # Train with augmentation
    print("\nAugmenting training data...")
    augmented_data = augment_training_data(training_data, target_per_intent=20)
    print(f"Original: {len(training_data)} examples")
    print(f"Augmented: {len(augmented_data)} examples")
    
    chatbot.train_intent_classifier(augmented_data, use_ensemble=True, optimize=True)
    
    # Test
    test_messages = ["hello", "I need help", "give me refund", "bye"]
    print("\nTesting chatbot:")
    for msg in test_messages:
        result = chatbot.get_detailed_analysis(msg)
        print(f"  User: {msg}")
        print(f"  Intent: {result['intent']} ({result['intent_confidence']:.2%})")
        print(f"  Bot: {result['response']}\n")