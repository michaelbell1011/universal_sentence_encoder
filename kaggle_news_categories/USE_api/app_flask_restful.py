from flask import Flask, request
from flask_restful import Resource, Api

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import json

# these two lines stop an OMP:error#15 exception when use_mtrx2vect_sim is run
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

 # create and finalize TF graph structure (finalizing optional but recommended).
# documentation: https://www.tensorflow.org/api_docs/python/tf/Graph#finalize

# Option to Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)

g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors (e.g. "list", or "array") of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  # import USE module from the internet. Can also be imported from local directory.
  # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
  embed = hub.Module("/Users/michaelbell/TFmodules")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)


# -- example request to TF to compute phrase vectors --
# for the request, session.run feeds input data into the graph through the placeholder.

# Input must be list of 1+ string(s).
# Output is a dictionary of phrase keys and vector array values

def use(str_list):
  embeddings = session.run(embedded_text, feed_dict={text_input: str_list})
  d={'phrases':str_list, 'vectors':embeddings}
  return d

# input list of avgd encoding dicts (e.g. [avg_use(), avg_use()]), and encoded test phrase dict (e.g. use())
def classify_use_avgs2vect(avgs_dict_list, test_use_dict):
    #   #access vect from encoding dict for test phrase
    test_vect = test_use_dict['vectors'].flatten()
    test_phrase = test_use_dict['phrases']

    # get list of avgd vects and compile list of group names
    groups = []
    vl = []
    for d in avgs_dict_list:
        vl.append(d['vect_avg'])
        groups.append(d['group_name'])
    # convert vect list to matrix
    mtrx = np.array(vl)
    # compute similarity
    results = np.inner(mtrx, test_vect)

    # compile df as output
    df = pd.DataFrame(data=results, index=groups, columns=test_phrase)
    # sort df descending
    sorted_df = df.sort_values(test_phrase, ascending=False)
    # # sorts df descending and filters to 'top n' results
    # df.nlargest(1, columns= test_phrase)
    return sorted_df


# API set up
# ========================================
app = Flask(__name__)
api = Api(app)

# root endpoint; will echo anything passed in POST body parameter
class HelloWorld(Resource):
    def get(self):
        return {
            'about': 'Hello World! Simple flask_restful application',
            'available routes': [" /(POST -d '{json}')",
                                "/classify(POST -d '{json}')"]
                }
# example request for root path:
# curl http://127.0.0.1:5000/

    def post(self):
        some_json = request.get_json()
        return {'you sent' : some_json}, 201
# test request with a POST body parameter:
# curl -H "Content-Type: application/json" -X POST -d '{"name":"Michael Bell", "age":28, "things I like": ["Colorado","eating"]}' http://127.0.0.1:5000/



class Classify(Resource):
    def post(self):
        requested_phrases = request.get_json() #see below format for request

        embeddings = use(requested_phrases['phrases'])
        num_embeds = len(embeddings['vectors'])

        # access the desired model
        with open("./model_store/big3_50training.json") as f:
            model = json.load(f)
            keys = list(model[0])
        # restore values as np.arrays
        for d in model:
            d['vect_avg'] = np.asarray(d['vect_avg'])
            d['sum_vec'] = np.asarray(d['sum_vec'])

        # make classifications
        classifications = np.array([])
        sim_scores = np.array([])
        for i in range(len(requested_phrases['phrases'])):
            # get corresponding headline and vector, and format as the output of USE() for input to classify()
            test_enc = {'phrases': [embeddings['phrases'][i]], 'vectors': embeddings['vectors'][i]}
            pred_df = classify_use_avgs2vect(model, test_enc)
            pred_str = pred_df.head(1).index[0]
            score = pred_df.head(1).iat[0,0]
            classifications = np.append(classifications, pred_str)
            sim_scores = np.append(sim_scores, score)

        return {'Requested phrases to classify': requested_phrases['phrases'],
                # 'number of requested phrases': len(requested_phrases['phrases']),
                # 'number of embedded vectors': num_embeds,
                # 'model keys': keys,
                'Classifications': classifications.tolist(),
                'Similarity scores': sim_scores.tolist()}, 201
# test classification request passing the input phrases as a parameter in the POST message body:
# curl -H "Content-Type: application/json" -X POST -d '{"phrases": ["Thanks Obama.", "Thanks Oprah!", "50 best ways to diet"]}' http://127.0.0.1:5000/classify

# add resource endpoint to api here
api.add_resource(HelloWorld, '/')
api.add_resource(Classify, '/classify') # endpoint takes in phrases as json-formatted array parameter in POST message body: -d '{"phrases": [...]'

if __name__ == '__main__':
    app.run(debug=True)