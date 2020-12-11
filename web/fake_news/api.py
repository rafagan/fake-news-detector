import os

from web.fake_news.parser import PredictParser
from web.main import app
from web.utils.decorators import auth, body
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score


@app.route('/', methods=['GET'])
def hello():
    return {'status': 'ok', 'payload': 'Hello!'}, 200


@app.route('/predict/', methods=['POST'])
@auth()
@body(PredictParser)
def predict(body):
    vocabulary_size = 5000
    neural_input_length = 20
    neural_net_data_path = '/app/neural_net/test.txt'

    ps = PorterStemmer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', body['text'])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # TODO: português
    review = ' '.join(review)
    corpus.append(review)

    bag_of_words = [one_hot(words, vocabulary_size) for words in corpus]
    X = np.array(pad_sequences(bag_of_words, padding='pre', maxlen=neural_input_length))

    return {'status': 'ok', 'payload': X.tolist()}, 200

    # model = Sequential()
    # model.load_weights(neural_net_data_path)
    # y = model.predict_classes(X)
    # return {'status': 'ok', 'payload': y}, 200
