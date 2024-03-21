from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

# API URL: https://goals-gameboard.azurewebsites.net

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/message', methods=['POST'])
@cross_origin()
def home():
    return jsonify({'message': 'Hello World!'}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    print("Running on port 5000")
