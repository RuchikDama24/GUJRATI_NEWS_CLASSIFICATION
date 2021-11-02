from flask import Flask, request, render_template
import pandas as pd
import nltk

from load_objects import *


app = Flask(__name__)


@app.route('/')
def get_landing_page():
    return render_template('index.html', error='*')


@app.route('/classify', methods=['POST', 'GET'])
def get_classification_result():
    if request.method == 'POST':
        text = request.form.get('text')
        
        # Handling empty form submission
        if text == '':
            return render_template('index.html', error='Field cannot be null')
    
        # Loading model objects stored when creating model from .ipynb file
        cv = load_count_vectorizer()
        le = load_label_encoder()
        tf_idf = load_tf_idf()
        clf = load_gujarathi_classifier()

        # Data preprocessing and classification task
        text_series = pd.Series([text])
        text = cv.transform(text_series)
        text = tf_idf.transform(text)
        classification_class = clf.predict(text)
        
        # Get original string label performing reverse transformation
        text = le.inverse_transform(classification_class)

        return render_template('result.html', result=text[0])


if __name__ == '__main__':
    app.run(threaded=False)
