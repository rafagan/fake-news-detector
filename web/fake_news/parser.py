from flask_restful import reqparse


class PredictParser:
    @staticmethod
    def parser():
        parser = reqparse.RequestParser(bundle_errors=True)

        # Primeira etapa
        parser.add_argument('text', type=str, required=True, help='text is required')
        return parser
