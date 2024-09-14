from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask! This is main page for deepfake app.'

@app.route('/test')
def model_train():
    return {
        "name": "Govinda Mandal",
        "email": "govinda4india@gmail.com"
    }

if __name__ == '__main__':
    app.run(debug=True)