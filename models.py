"""
Models Module

This module provides machine learning functionality for the Lead Scorer project.

It includes the LeadClassifier class for training and predicting lead categories,
as well as utility functions for model management, evaluation, and batch processing.

Key Components:
- LeadClassifier: Main class for text classification models
- Model training and persistence functions
- Evaluation and comparison utilities
- Text preprocessing with lemmatization

Supported Models:
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Ensemble (Voting Classifier)

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import os

# Set nltk data path to avoid path issues on Windows
nltk.data.path.insert(0, os.path.join(os.getcwd(), 'nltk_data'))

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatize_text(text):
    """
    Preprocess text by lemmatizing words and removing stop words.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Processed text with lemmatized words and stop words removed.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

class LeadClassifier:
    """Machine learning model for classifying leads into Hot, Warm, or Cold categories."""
    
    def __init__(self, model_path="lead_logistic.pkl"):
        self.model_path = model_path
        self.model = None
        
    def build_model(self, model_type='logistic'):
        """Create a pipeline with TF-IDF vectorizer and classifier.

        model_type options: 'logistic', 'svm', 'naive_bayes', 'ensemble'
        """
        vectorizer = TfidfVectorizer(
            max_features=100,  # Reduced to reduce overfitting
            preprocessor=lemmatize_text,  # Lemmatization and stop word removal
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Require term in at least 2 docs
            max_df=0.9  # Remove very common words
        )

        if model_type == 'ensemble':
            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
            nb = MultinomialNB(alpha=0.1)
            svm = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
            classifier = VotingClassifier(estimators=[('lr', lr), ('nb', nb), ('svm', svm)], voting='hard')
        elif model_type == 'logistic':
            classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
        elif model_type == 'svm':
            classifier = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
        else:  # naive_bayes
            classifier = MultinomialNB(alpha=0.1)

        self.model = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        return self.model

    def augment_data(self, df):
        """Simple data augmentation by paraphrasing messages."""
        augmented = []
        for _, row in df.iterrows():
            msg = str(row['Message'])
            # Replace common terms with synonyms
            aug1 = msg.replace('demo', 'demonstration').replace('pricing', 'cost').replace('quote', 'estimate')
            if aug1 != msg:
                augmented.append({
                    'Message': aug1,
                    'Label': row['Label'],
                    'Job title': row.get('Job title', ''),
                    'Company': row.get('Company', '')
                })
            # Add another variation
            aug2 = msg.replace('need', 'require').replace('want', 'desire')
            if aug2 != msg and aug2 != aug1:
                augmented.append({
                    'Message': aug2,
                    'Label': row['Label'],
                    'Job title': row.get('Job title', ''),
                    'Company': row.get('Company', '')
                })
        df_aug = pd.DataFrame(augmented)
        df = pd.concat([df, df_aug], ignore_index=True)
        print(f"Augmented data: added {len(df_aug)} samples")
        return df
    
    def train(self, csv_path="Data/csvfile.csv"):
        """Train the model on the CSV data."""
        from lead_scorer import judge_lead

        # Read the CSV file
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return False

        # Try common encodings if default fails
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except Exception as e:
                print(f"Error reading CSV with fallback encoding: {e}")
                return False
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False

        # Check if required columns exist
        if 'Message' not in df.columns:
            print("Error: CSV must have 'Message' column")
            return False

        # Remove rows with missing Message
        df = df.dropna(subset=['Message'])

        # Data augmentation to reduce overfitting
        df = self.augment_data(df)

        if len(df) < 2:
            print("Error: Not enough data to train the model (minimum 2 samples)")
            return False

        # Compute labels using judge_lead
        y = []
        for idx, row in df.iterrows():
            _, label, _, _ = judge_lead(row['Message'], row.get('Job title', ''), row.get('Source', ''), row.get('Company', ''))
            y.append(label)
        y = np.array(y)

        # Extract features: combine message with job title and company for richer context
        X = []
        for idx, row in df.iterrows():
            msg = row['Message']
            job = row.get('Job title', '')
            company = row.get('Company', '')
            combined = f"{msg} {job} {company}".strip()
            X.append(combined)
        X = np.array(X)

        # Build and train the model
        self.build_model()
        self.model.fit(X, y)

        print(f"Model trained on {len(df)} samples")

        # Print cross-validation metrics
        cv_accuracy = cross_val_score(self.model, X, y, cv=5, scoring='accuracy').mean()
        cv_f1 = cross_val_score(self.model, X, y, cv=5, scoring='f1_weighted').mean()
        print(f"CV Accuracy: {cv_accuracy:.2%}")
        print(f"CV F1-Score (weighted): {cv_f1:.4f}")

        return True
    
    def predict(self, message, job_title='', company=''):
        """Predict the label for a new message, optionally including job title and company."""
        if self.model is None:
            print("Error: Model not trained. Call train() first.")
            return None, 0.0

        combined = f"{message} {job_title} {company}".strip()
        prediction = self.model.predict([combined])[0]
        probability = max(self.model.predict_proba([combined])[0])

        return prediction, probability
    
    def predict_batch(self, messages):
        """Predict labels for multiple messages."""
        if self.model is None:
            print("Error: Model not trained. Call train() first.")
            return None
        
        predictions = self.model.predict(messages)
        probabilities = np.max(self.model.predict_proba(messages), axis=1)
        
        return predictions, probabilities
    
    def save(self):
        """Save the trained model to a file."""
        if self.model is None:
            print("Error: No model to save. Train the model first.")
            return False
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")
        return True
    
    def load(self):
        """Load a previously trained model from a file."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_path}")
        return True
    
    def get_model_info(self):
        """Get information about the trained model."""
        if self.model is None:
            return "Model not trained"
        
        return {
            'type': type(self.model).__name__,
            'vectorizer': 'TF-IDF',
            'classifier': type(self.model.named_steps['classifier']).__name__,
            'labels': ['Hot', 'Warm', 'Cold']
        }


# Convenience functions
def create_and_train_classifier(csv_path="Data/csvfile.csv"):
    """Create, train, and save a new classifier."""
    classifier = LeadClassifier()
    if classifier.train(csv_path):
        classifier.save()
        return classifier
    return None


def load_classifier(model_path="lead_logistic.pkl"):
    """Load a previously trained classifier."""
    classifier = LeadClassifier(model_path)
    if classifier.load():
        return classifier
    return None


def train_and_save_models(csv_path="Data/csvfile.csv", output_prefix="lead_"):
    """Train multiple model types and save each as a separate pickle.

    Saves files like `{output_prefix}logistic.pkl`, `{output_prefix}svm.pkl`, etc.
    Returns list of saved paths.
    """
    from lead_scorer import judge_lead

    if not os.path.exists(csv_path):
        print(f"Training CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV for training: {e}")
        return []

    if 'Message' not in df.columns:
        print("CSV must contain 'Message' column")
        return []

    df = df.dropna(subset=['Message'])

    # Use existing labels if available, else compute using judge_lead
    if 'Label' in df.columns:
        y = df['Label'].values
        print("Using existing 'Label' column for training.")
    else:
        print("No 'Label' column found, computing labels using rule-based judge_lead.")
        y = []
        for idx, row in df.iterrows():
            _, label, _, _ = judge_lead(row['Message'], row.get('Job title', ''), row.get('Source', ''), row.get('Company', ''))
            y.append(label)
        y = np.array(y)

    # Extract features: combine message with job title and company for richer context (consistent with individual training)
    X = []
    for idx, row in df.iterrows():
        msg = row['Message']
        job = row.get('Job title', '')
        company = row.get('Company', '')
        combined = f"{msg} {job} {company}".strip()
        X.append(combined)
    X = np.array(X)

    saved = []
    for mtype in ('logistic', 'svm', 'naive_bayes', 'ensemble'):
        lc = LeadClassifier(model_path=f"{output_prefix}{mtype}.pkl")
        lc.build_model(model_type=mtype)
        try:
            lc.model.fit(X, y)
            lc.save()
            saved.append(lc.model_path)
            print(f"Trained and saved {mtype} -> {lc.model_path}")
        except Exception as e:
            print(f"Failed to train/save {mtype}: {e}")

    return saved


def _predict_with_confidence(pipeline, messages):
    """
    Predict labels and confidence scores for a list of messages using a trained pipeline.

    Args:
        pipeline: Trained sklearn pipeline with predict and predict_proba methods.
        messages (list): List of text messages to classify.

    Returns:
        tuple: (predictions, top_probabilities, probabilities_list)
            - predictions: List of predicted labels
            - top_probabilities: List of highest probability for each prediction
            - probabilities_list: List of dicts with class probabilities
    """
    import numpy as _np
    preds = []
    tops = []
    probs_list = []
    for m in messages:
        try:
            if hasattr(pipeline, 'predict_proba'):
                p = pipeline.predict_proba([m])[0]
                classes = getattr(pipeline, 'classes_', None)
                preds.append(pipeline.predict([m])[0])
                tops.append(float(_np.max(p)))
                probs_list.append(dict(zip(classes, p)) if classes is not None else {})
                continue

            if hasattr(pipeline, 'decision_function'):
                df = pipeline.decision_function([m])[0]
                arr = _np.array(df)
                if arr.ndim == 0:
                    prob_pos = 1.0 / (1.0 + _np.exp(-float(arr)))
                    preds.append(pipeline.predict([m])[0])
                    tops.append(float(prob_pos))
                    probs_list.append({})
                else:
                    exps = _np.exp(arr - _np.max(arr))
                    probs = exps / exps.sum()
                    classes = getattr(pipeline, 'classes_', None)
                    preds.append(pipeline.predict([m])[0])
                    tops.append(float(_np.max(probs)))
                    probs_list.append(dict(zip(classes, probs)) if classes is not None else {})
                continue

            # fallback
            preds.append(pipeline.predict([m])[0])
            tops.append(None)
            probs_list.append({})
        except Exception:
            preds.append(None)
            tops.append(None)
            probs_list.append({})

    return preds, tops, probs_list


def list_saved_model_paths(pattern='lead_*.pkl'):
    """
    Find and return paths to saved model pickle files matching the given pattern.

    Args:
        pattern (str): Glob pattern to match model files (default: 'lead_*.pkl').

    Returns:
        list: List of file paths matching the pattern.
    """
    import glob as _glob
    return _glob.glob(pattern)


def evaluate_saved_models(csv_path=None, sample_messages=None):
    """Load saved models and produce separate outputs per model.

    - If `csv_path` is provided and exists, prints classification report per model.
    - If `sample_messages` is provided, prints per-message predictions and probabilities per model.
    Returns a dict mapping model_path -> results.
    """
    from sklearn.metrics import classification_report as _cr
    results = {}
    paths = list_saved_model_paths()
    if not paths:
        print("No saved model files found matching 'lead_*.pkl'")
        return results

    # load dataset if provided
    X = None
    y = None
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='cp1252')
            except Exception as e:
                print(f"Could not read csv_path={csv_path}: {e}")
                df = None
        except Exception as e:
            print(f"Could not read csv_path={csv_path}: {e}")
            df = None

        if df is not None and {'Message', 'Label'}.issubset(df.columns):
            df = df.dropna(subset=['Message', 'Label'])
            X = df['Message'].values
            y = df['Label'].values

    # use sample messages if given, else a tiny default set
    if sample_messages is None:
        sample_messages = [
            "Please share pricing and demo for your solution.",
            "Just browsing your website, looks interesting.",
            "I am a student interested in learning about AI."
        ]

    for p in paths:
        try:
            with open(p, 'rb') as f:
                pipeline = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        model_res = {}

        # If we have labeled data, compute classification report
        if X is not None and y is not None:
            try:
                preds = pipeline.predict(X)
                report = _cr(y, preds)
                model_res['classification_report'] = report
                print(f"\n=== Report for {p} ===")
                print(report)
            except Exception as e:
                model_res['classification_report'] = f"error: {e}"
                print(f"Could not compute report for {p}: {e}")

        # Predictions for sample messages
        try:
            preds, tops, probs_list = _predict_with_confidence(pipeline, sample_messages)
            print(f"\n=== Sample predictions for {p} ===")
            for i, msg in enumerate(sample_messages):
                top_str = f"{tops[i]:.1%}" if tops[i] is not None else "N/A"
                print(f"- Message: {msg}")
                print(f"  Prediction: {preds[i]}  TopConf: {top_str}")
                if probs_list[i]:
                    probs_str = ', '.join([f"{k}:{v:.1%}" for k, v in sorted(probs_list[i].items(), key=lambda x: -x[1])])
                    print(f"  Probabilities: {probs_str}")
        except Exception as e:
            print(f"Could not predict samples for {p}: {e}")

        results[p] = model_res

    return results


def load_all_saved_models(pattern='lead_*.pkl'):
    """Load all saved models matching the pattern and return list of dicts with 'name' and 'pipeline'."""
    paths = list_saved_model_paths(pattern)
    models = []
    for p in paths:
        try:
            with open(p, 'rb') as f:
                pipeline = pickle.load(f)
            # Extract name from path, e.g., 'lead_classifier_logistic.pkl' -> 'logistic'
            name = os.path.basename(p).replace('lead_classifier_', '').replace('.pkl', '')
            models.append({'name': name, 'pipeline': pipeline})
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return models


# Main block for direct execution
if __name__ == "__main__":
    print("=== Lead Classifier Training ===\n")
    
    print("Training all model types (logistic, svm, naive_bayes)...")
    saved_paths = train_and_save_models(csv_path="Data/csvfile.csv", output_prefix="lead_classifier_")
    
    if not saved_paths:
        print("Failed to train models.")
        exit(1)
    
    print(f"\nSuccessfully trained and saved {len(saved_paths)} models:")
    for p in saved_paths:
        print(f"  - {p}")
