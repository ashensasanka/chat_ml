from flask import Flask, request, jsonify
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from model import predict_class,get_response
intents = json.loads(open('content.json').read())
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))
X = df['comment_text']
vectorizer.adapt(X.values)


app = Flask(__name__)

@app.route('/api', methods = ['GET'])
def returnascii():
    inputchr = str(request.args['query'])
    ints = predict_class(inputchr)
    res = get_response(ints,intents)
    d = {}
    d['output'] = str(res)
    print(res)
    return d



if __name__ =="__main__":
    app.run()
