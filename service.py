from flask import Flask
from flask_restful import Resource, Api, reqparse

from inference import TargetPredictor

app = Flask(__name__)
api = Api(app)

get_parser = reqparse.RequestParser()
get_parser.add_argument('text')


class TextReadability(Resource):

    def __init__(self):
        self.target_predictor = TargetPredictor()

    def get(self):
        args = get_parser.parse_args()
        if args['text']:
            return {'score': self.target_predictor.predict(args['text'])}
        else:
            return {'error': 'No text provided'}


api.add_resource(TextReadability, '/get_text_readability')

if __name__ == '__main__':
    app.run()
