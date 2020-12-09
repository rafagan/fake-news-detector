from web.fake_news.parser import PredictParser
from web.main import app
from web.utils.decorators import auth, body


@app.route('/', methods=['GET'])
def hello():
    return {'status': 'ok', 'payload': 'Hello!'}, 200


@app.route('/predict/', methods=['POST'])
@auth()
@body(PredictParser)
def predict(body):
    return {'status': 'ok', 'payload': body}, 200
