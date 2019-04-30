import json
import numpy as np

with open("./model_store/big3_50training_v2.json") as f:
    model = json.load(f)

# restore values as np.arrays
for d in model:
    d['vect_avg'] = np.asarray(d['vect_avg'])
    d['sum_vec'] = np.asarray(d['sum_vec'])


    # THESE ARE NO LONGER SAVED IN THE MODEL
    # d['averaged_phrases'] = np.asarray(d['averaged_phrases'])
    # d['phrase_vects'] = np.asarray(d['phrase_vects'])

    print(type(d['vect_avg']))
    print(d['vect_avg'].shape)


print(model[0]['count'])