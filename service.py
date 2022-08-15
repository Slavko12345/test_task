from flask import Flask
from flask_restful import Resource, Api

from inference import TargetPredictor

app = Flask(__name__)
api = Api(app)


class TextReadability(Resource):

    def __init__(self):
        self.target_predictor = TargetPredictor()

    def get(self, text):
        return {'score': self.target_predictor.predict(text)}


api.add_resource(TextReadability, '/get_text_readability/<text>')

if __name__ == '__main__':
    app.run()
