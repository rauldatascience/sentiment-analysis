import pickle

from os import environ
from os.path import join, dirname
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from config import FEATURE_EXTRACTOR_FILEPATH, CLASSIFIER_FILEPATH, LABELS

from config import DATA_FILEPATH

from train import main

app = Flask(__name__)

with open(FEATURE_EXTRACTOR_FILEPATH, 'rb') as infile:
    app.feature_extractor = pickle.load(infile)

with open(CLASSIFIER_FILEPATH, 'rb') as infile:
    app.classifier = pickle.load(infile)

def reply_success(data):
    response = jsonify({
        "data": data
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

def reply_error(code, message):
    response = jsonify({
        "error": {
            "code": code,
            "message": message
        }
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

@app.route("/")
def index():
    return "<h1>Sentiment Analysis API using Flask</h1>"

@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "GET":
        text = request.args.get("text", None)
    elif request.method == "POST":
        json_req = request.get_json()
        text = json_req["text"]
    else:
        return reply_error(code=400, message="Supported method is 'GET' and 'POST'")

    if text:
        # IMPORTANT: Use [text] because sklearn vectorizer expects an iterable as the input
        # IMPORTANT: classifier.predict returns an array, so get the first element
        label = app.classifier.predict(app.feature_extractor.transform([text]))[0]

        return reply_success(data={
            'text': text,
            'sentiment': LABELS[label]
        })

    return reply_error(code=400, message="Text is not specified")

## Start modify build endpoint /feedback

# make a second input funtion
def reply_success_2(data_2):
    response = jsonify({
        "data": data_2
    })
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# route the /clssify/feedback with only POST method
@app.route("/classify/feedback", methods=["POST"])
def feedback():
    if request.method == "POST":
        json_req = request.get_json()
        # receiving 2 elements, text and sentiment as a keys
        text_2 = json_req["text"]
        text_3 = json_req["sentiment"]
    else:
        return reply_error(code=400, message="Supported method is'POST'")

    # Read data/tweets_100k/positive.txt again
    with open(DATA_FILEPATH + "/positive.txt", "r") as infile:
        positive_tweets = infile.readlines()
    
    # Read data/tweets_100k/negative.txt again
    with open(DATA_FILEPATH + "/negative.txt", "r") as infile:
        negative_tweets = infile.readlines()

    # Perform a check whether the input text object is contained in the file positive.txt or negative.txt 
    # if the input text object is in negative_tweets or positive_tweets, we will do nothing just response
    if text_2 in negative_tweets or text_2 in positive_tweets:
        return reply_success_2(data_2={
            'text': text_2,
            'sentiment': text_3,
            "msg": "We have it already!"            
        })
    # if not exist in positive.txt or negative.txt
    # The incoming text will store to the count_new_data_added.txt
    # with this approach, we can investigate what text objects were recently added
    # we also trying to filter if the incoming text object has a positive sentiment so we store to the positive.txt
    # and vice versa.

    else:
        
        with open(DATA_FILEPATH + "/count_new_data_added.txt", "a") as file_object:
                file_object.write(str(text_2) + ' \n')

        if text_3 == "positive":
            with open(DATA_FILEPATH + "/positive.txt", "a") as file_object:
                file_object.write(' \n' + str(text_2))
        else:
            with open(DATA_FILEPATH + "/negative.txt", "a") as file_object:
                file_object.write(' \n' + str(text_2))

        return reply_success_2(data_2={
            'text': text_2,
            'sentiment': text_3,
            "msg": "Your feedback is well received!"   
        })
    return reply_error(code=400, message="Text is not specified")
    
    # at last, we are trying to re-train the model whenever 10 data is added
    if len(positive_tweets + negative_tweets) % 10 == 0:
        return train.main()

if __name__ == "__main__":
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    port = int(environ.get("PORT"))
    debug = environ.get("DEBUG")

    if debug == "True":
        app.run(threaded=True, port=port, debug=True)
    else:
        app.run(threaded=True, port=port, debug=False)
