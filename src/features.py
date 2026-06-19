from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def extract_features(train_data, test_data, method="tfidf", max_features=5000):
    """
    Transforms clean text into numerical feature spaces using either CountVectorizer or TfidfVectorizer.
    """
    if method == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    else:
        raise ValueError("Invalid representation method specified. Choose 'bow' or 'tfidf'.")

    X_train_vec = vectorizer.fit_transform(train_data)
    X_test_vec = vectorizer.transform(test_data)

    return X_train_vec, X_test_vec, vectorizer
