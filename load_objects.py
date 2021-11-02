import pickle


def load_count_vectorizer():
    with open('../objects/gujarathi-vectorizer.pickle', 'rb') as file:
        cv = pickle.load(file)

    return cv


def load_label_encoder():
    with open('../objects/gujarathi-encoder.pickle', 'rb') as file:
        le = pickle.load(file)

    return le


def load_tf_idf():
    with open('../objects/gujarathi-tf-idf.pickle', 'rb') as file:
        tfidf = pickle.load(file)

    return tfidf


def load_gujarathi_classifier():
    with open('../objects/gujurathi-classifier.pickle', 'rb') as file:
        model = pickle.load(file)

    return model
