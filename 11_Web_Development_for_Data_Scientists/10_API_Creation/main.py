from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    # Some ML Model
    data = {"Output": 45, "Accuracy": 0.98}
    return jsonify(data), 200

app.run(debug=True)