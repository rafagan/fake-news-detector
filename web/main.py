from flask import Flask
from flask_cors import CORS

from web.utils import DecimalEncoder

app = Flask(__name__)
app.json_encoder = DecimalEncoder
CORS(app)

# noinspection PyUnresolvedReferences
import web.fake_news.api
