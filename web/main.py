import json
import uuid
from flask import Flask

app = Flask(__name__)


@app.route("/")
def index():
    return ('<a href="https://docs.google.com/document/d/1dbq_X07W1VE3jCLwL'
            'Y8qFrVpsULZaK_S5pJqz3R4Cmg/edit?usp=sharing">API Spec</p>')


@app.route("/recognize", methods=["POST", "GET"])
def recognize():
    return json.dumps({
        "img_id": str(uuid.uuid4()),
        "candidates": [
            {
                "label_id": str(uuid.uuid4()),
                "label": "Milk",
                "score": 0.8,
            },
        ]
    })


@app.route("/confirm", methods=["POST", "GET"])
def confirm():
    return json.dumps({
        "status": "ok"
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
