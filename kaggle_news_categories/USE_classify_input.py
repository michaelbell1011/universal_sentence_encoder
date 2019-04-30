# import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
# import pandas as pd
import json

import computations as comp


# these two lines stop an OMP:error#15 exception when use_mtrx2vect_sim is run
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# expands the JSONEncoder to convert np.arrays to lists
# from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# function to write model to model store directory as json file
def modelJSON(m):
    file_name = input("File name for model (.json is added automatically):")
    with open('./model_store/'+file_name+'.json', 'w') as f:
        json.dump(m, f, cls=NumpyEncoder, indent=2)
    f.close()

# =================================================

def main():
    """Opens model object from file and uses as reference for classifying user input phrase"""
    # comp.Hello()

    # load model.json from model store-- currently hard-coded
    with open("./model_store/big3_50training.json") as f:
        model = json.load(f)

    # restore values as np.arrays
    for d in model:
        d['vect_avg'] = np.asarray(d['vect_avg'])
        d['sum_vec'] = np.asarray(d['sum_vec'])


    # testing classification of a single example phrase
    # this version will return all the categories in the model and their similarity scores in descending order.
    p= [input('Test headline:')]
    input_phrase = comp.use(p)
    print(input_phrase.keys())

    result_df = comp.classify_use_avgs2vect(model, input_phrase)
    print(result_df)

    pred_str = result_df.head(1).index[0]
    print(f'top classification: {pred_str}')

    score = float(result_df.head(1).iat[0,0])
    print(f'sim scores is: {score}')
    print(type(score))

if __name__ == '__main__':
    main()