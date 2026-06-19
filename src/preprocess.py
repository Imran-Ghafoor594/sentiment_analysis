import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads configure karna
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Applies rigorous text preprocessing pipeline to the incoming legal notices.
    Steps: HTML removal, Lowercasing, Punctuation cleaning, Tokenization, Stopword extraction, Lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML tag removal
    text = re.sub(r'<[^>]+>', '', text)

    # 2. Lowercasing
    text = text.lower()

    # 3. Punctuation & special character removal
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenisation & Stopword removal
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # 5. Lemmatisation (Reduces words to dictionary roots)
    lemmatizer = WordNetLemmatizer()
    processed_words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(processed_words)
