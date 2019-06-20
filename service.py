from flask import Flask, request
from flask_restful import Api, Resource
import logging
import ocr_detect
from flask_cors import CORS
from flask import send_from_directory
import os
 
static_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')



logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p',
        level=logging.INFO)

app = Flask(__name__)
CORS(app)
api = Api(app)


class OcrOneRegion(Resource):

    def get(self):
        # Create two threads as follows
        try:
            input_url = request.args.get('input_url')
            x = request.args.get('x')
            y = request.args.get('y')
            height = request.args.get('height')
            width = request.args.get('width')

            print(input_url)

            result = ocr_detect.recognize_by_cor(input_url, int(x), int(y), int(height), int(width))

            message = {"message": result}
            code = 200
        except:
            logging.warning("Error: ", exc_info=True)
            message = {"message": ("call fail ")}
            code = 401

        return message, code


class OcrKeyDetectService(Resource):
    
    def get(self):
        # Create two threads as follows
        try:
            input_url = request.args.get('input_url')
            print(input_url)

            result = ocr_remote_pdf.process_url_image(input_url)

            message = {"message": result}
            code = 200
        except:
            logging.warning("Error: ", exc_info=True)
            message = {"message": ("call fail ")}
            code = 401

        return message, code

api.add_resource(OcrKeyDetectService, '/ocr_key_detect')
api.add_resource(OcrOneRegion,'/ocr_region_detect')

@app.route('/static_ocr/<path:path>', methods=['GET'])
def serve_file_in_dir(path):
    if not os.path.isfile(os.path.join(static_file_dir, path)):
        message = {"message": ("Not Found")}
        code = 404
        return message, code
 
    return send_from_directory(static_file_dir, path)

app.run(host= '0.0.0.0', port=8081)

