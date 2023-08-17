from flask import Flask, jsonify, request
from detect import ShotDetector  # Update the import path
import requests
class Controller:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.route('/match-analysis', methods=['POST'])(self.get_match_analysis)  # Change to POST

    def run(self):
        self.app.run(host='localhost', port=5000)

    def get_match_analysis(self):
        data = request.json  # Get the JSON data from the POST request
        detect = ShotDetector(data['VideoUrl'])
        result = detect.run()
        print(f'Shot detection result: {result}')
        return jsonify(result)
    
if __name__ == "__main__":
    controller = Controller()
    controller.run()