import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
import json
import time

# NOTE the script currently filters the dataset to the "big3" classes on line 207:
# 'POLITICS' |  'ENTERTAINMENT' | 'WELLNESS'

# these two lines stop an OMP:error#15 exception
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
  # import USE module
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
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

# ---------------------------
# Helper functions

# compute an average vector for a group of encoded phrases.
# input is a dictionary of multiple encoded phrases passed to USE()
# output is a new dictionary with the name of the group, list of included phrases, and the averaged vector
def use_avg(use_dict):
    # name the phrase group via user input prompt
    group_name = input("What is the name for this group of phrases?")
    print(f'Phrase group saved as: {group_name}')

    # access vectors in embedding dicitionary and compute average
    matrix = use_dict['vectors']
    sum_vec = matrix.sum(axis=0)
    avg = sum_vec / matrix.shape[0]

    # store results as dictionary
    phrase_group = use_dict['phrases']
    results_dict = {'group_name': group_name, 'vect_avg': avg, 'averaged_phrases': phrase_group, 'phrase_vects': matrix}
    print(f'Keys for averaged vector calculation: {list(results_dict)}\n')
    return results_dict


# # functions to compute Similarity Scores via vector inner "dot" product:

# compute scores for 1+ reference phrases * test phrase vector
# might be helpful for checking covariance within group sample phrases
def use_mtrx2vect_sim(group_use_dict, test_use_dict):
    # access matrix of vects from encoding dict for phrase group
    ref_mtrx = group_use_dict['vectors']
    # access vect from encoding dict for test phrase
    test_vect = test_use_dict['vectors'].flatten()
    # calculate similarity scores between the test and reference vectors
    scores = np.inner(ref_mtrx, test_vect)
    # log message and return scores array
    print(
        f"Calculating similarity scores for Reference Phrases: {group_use_dict['phrases'][:5]}... \nand: {test_use_dict['phrases']}...")
    return scores


# compute score for 1 phrase-group average vector * test phrase vector
def use_avg2vect_sim(avg_vect_dict, test_use_dict):
    # access intent avg vect from dict
    single_ref_vect = avg_vect_dict['vect_avg']
    # access vect from encoding dict for test phrase
    test_vect = test_use_dict['vectors'].flatten()
    # calculate similarity score between the two vectors
    score = np.inner(single_ref_vect, test_vect)
    # log message and return score float
    print(
        f"Calculating similarity score for Group: {avg_vect_dict['group_name']} and Test Phrase: {test_use_dict['phrases']}...")
    return score


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
# ---------------------------
# Evaluation functions

# evaluate the variance of similarity scores within a phrase group * it's average
# pass in the encoded dict of a phrase-group, and that group's avgeraged vector dict
def eval_matrx2avg_sim(group_use_dict, avg_vect_dict):
    # re-compile avg dict for input to use_mtrx2vect_sim()
    name = str(f"{avg_vect_dict['group_name']} group average vector")
    v = avg_vect_dict['vect_avg']
    #   simple_avg_enc = {name:v}
    simple_avg_enc = {'phrases': name, 'vectors': v}

    # evaluate similarity of all group sample phrases to group Avg Vect as the "test" phrase
    scores = use_mtrx2vect_sim(group_use_dict, simple_avg_enc)

    print(f'Phrase group similarity scores to their avgerage vector are {scores}')
    print(f'Mean: {scores.mean()}')
    print(f'std: {scores.std()}')
    print(f'max: {scores.max()}')
    print(f'min: {scores.min()}')
    return scores

# TOO INEFFICIENT TO SAVE ALL TRAINING PHRASES AND THEIR VECTORS TO THE MODEL OBJECT, SO THIS FUNCTION IS MOOT NOW
# # function to evaluate the classification reliability for all phrases in a list of averaged phrase groups
# # pass in list of use_avg dicts
# def eval_classifer(l_o_use_avg_dicts):
#     # compile one df using df.append() from list of avg dicts
#     # instantiate empty df with columns 'averaged_phrases' and 'phrase_vects'
#     main_df = pd.DataFrame(columns=['phrase', 'vector', 'group_name'])
#
#     # update l_o_avgs below
#     for d in l_o_use_avg_dicts:
#         df_data = dict(phrase=d['averaged_phrases'], vector=list(d['phrase_vects']))
#         df = pd.DataFrame(data=df_data)
#         df['group_name'] = d['group_name']
#         main_df = main_df.append(df, ignore_index=True)
#
#     # make classification predictions and add to main_df
#     pred_group_list = []
#     for i in range(len(main_df)):
#         test_enc = {'phrases': [main_df['phrase'][i]], 'vectors': main_df['vector'][i]}
#         # update l_o_avgs below
#         pred_df = classify_use_avgs2vect(l_o_use_avg_dicts, test_enc)
#         pred_str = pred_df.head(1).index[0]
#         pred_group_list.append(pred_str)
#     main_df['pred_group'] = pred_group_list
#
#     # calculate accuracy and add to main_df
#     a = []
#     for i in range(len(main_df['group_name'])):
#         if main_df['group_name'].iloc[i] == main_df['pred_group'].iloc[i]:
#             a.append('correct')
#         else:
#             a.append('wrong')
#     main_df['accuracy'] = a
#
#     print(main_df['accuracy'].value_counts())
#     return main_df


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

# # =======================================================

# import data and filter to classes of interest
# made copy of data directory in USE_local_imps directory
data = pd.read_json(path_or_buf="data/News_Category_Dataset_v2.json", lines=True)
# remove duplicated headlines
data.drop_duplicates(subset='headline', inplace=True)

# this block is not needed since this class is filtered out below
#'THE WORLDPOST' and 'WORLDPOST' are the same category, merging them here.
# data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

# filter the dataset to just relevant columns for groups of interest
filters = ((data.category == 'POLITICS') | (data.category == 'ENTERTAINMENT') | (data.category == 'WELLNESS'))
filtered_data = data.loc[filters,['category','headline']]

# split orig data DF into train and test series
x= filtered_data['headline']
y= filtered_data['category']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=30, stratify=y)

print(f'number of test phrases: {len(x_test)}')
print(f'number of training phrases: {len(x_train)}')
print(f'proportion of held-out test data: {len(x_test)/len(filtered_data)}')
print(f'Total size of dataset: {len(x_test) + len(x_train)}')


# # =======================================================
# Model Training + Accuracy Evaluation
print('=== Model Training =========')

# COPIED FROM TEST SCRIPT ===============
# instantiate empty arrays to append the encoded data; maintains order after groupby operation
encd_headlines = np.array([])
encd_categories = np.array([])
encd_vects_list = []
#  ======================================

# re-combine x and y training series into DF to group class phrases
d={'headline':x_train, 'category':y_train}
train_df = pd.DataFrame(data= d)

# I discovered that USE has better performance when calls are looped in batches
# so this implementation encodes headlines one category at a time.

# training loop
training_start = time.time()
model = []

for g in filtered_data.category.unique():
    group_name = g

    # COPIED FROM TEST SCRIPT ===============
    # get the filtered categories and append to array
    categories = np.array(train_df.loc[train_df.category==g , 'category'])
    encd_categories = np.append(encd_categories, categories)
    #  ======================================

    # get the filtered headlines, encode them, and append 2d array to a list
    headlines = np.array(train_df.loc[train_df.category==g,'headline'])
    enc_train_start = time.time()
    encs = use(headlines)
    enc_train_end = time.time()
    print(f'Encoding {len(headlines)} test phrases for {g} took {enc_train_end - enc_train_start} seconds')
    print(f'encoding rate: {len(headlines) / (enc_train_end - enc_train_start)} phrases/sec')

    # COPIED FROM TEST SCRIPT ===============
    encd_headlines = np.append(encd_headlines, encs['phrases'])
    encd_vects_list.append(encs['vectors'])
    #  ======================================

    # compute average vector
    matrix = encs['vectors']
    sum_vec = matrix.sum(axis=0)
    avg = sum_vec / matrix.shape[0]

    # store results as dictionary
    # CHANGED WHAT IS STORED IN THE MODEL
    # results_dict = {'group_name': group_name, 'vect_avg': avg, 'averaged_phrases': headlines, 'phrase_vects': matrix}
    results_dict = {'group_name': group_name, 'vect_avg': avg, 'sum_vec': sum_vec, 'count': matrix.shape[0]}
    # add avg obj to 'model' list
    model.append(results_dict)


print(f'Model training finished. Number of included classes: {len(model)}')
training_end = time.time()
print(f'model training took: {training_end - training_start} seconds')


# # evaluate model accuracy classifying training data

# THIS LINE IS MOOT SINCE I CHANGED WHAT INFO IS SAVED IN THE MODEL. REPLACED WITH THE BELOW BLOCK FROM TEST SCRIPT.
# train_reliability = eval_classifer(model)

# COPIED FROM TEST SCRIPT ======================================
# this was the only way I could figure out how to compile a 2d array from the multiple encodings starting from an empty variable
encd_vects = np.concatenate(encd_vects_list, axis=0)

# classify the test headlines using the loaded model
predictions = np.array([])
for i in range(len(encd_headlines)):
    # get corresponding headline and vector, and format as the output of USE() for input to classify()
    test_enc = {'phrases': [encd_headlines[i]], 'vectors': encd_vects[i]}
    pred_df = classify_use_avgs2vect(model, test_enc)
    pred_str = pred_df.head(1).index[0]
    predictions = np.append(predictions, pred_str)

calssification_end = time.time()
print(f'classification of training data took: {calssification_end - training_end} seconds')

# compile results df
d = {'headline': encd_headlines, 'category': encd_categories, 'prediction': predictions}
results_df = pd.DataFrame(data= d)

# calculate accuracy for each df row and add results as column
a = []
for i in range(len(results_df.headline.values)):
    if results_df['category'].iloc[i] == results_df['prediction'].iloc[i]:
        a.append('correct')
    else:
        a.append('wrong')
results_df['accuracy'] = a

accuracy_computation_end = time.time()
print(f'training classification accuracy calculation took: {accuracy_computation_end - calssification_end} seconds')

# print results
print(results_df['accuracy'].value_counts())
print(len(results_df))
results_counts = results_df['accuracy'].value_counts()
print(f"model training classification accuracy: {results_counts['correct']/sum(results_counts)}")

testing_end = time.time()
print(f'Total evaluation time for training data: {testing_end - training_end}')
#  ======================================
# End of Model Training

# # =======================================================
# Model Testing
print('=== Model Testing =========')

# I discovered that USE has better performance when calls are looped in batches
# so this implementation encodes headlines one category at a time.

# instantiate empty arrays to append the encoded data; maintains order after groupby
encd_headlines = np.array([])
encd_categories = np.array([])
encd_vects_list = []

# re-combine the series for the headlines and their category labels into DF to perform batching
d={'headline':x_test, 'category':y_test}
test_df = pd.DataFrame(data= d)

# encode test phrases batched by their category label
testing_start = time.time()
for g in filtered_data.category.unique():
    # get the filtered categories and append to array
    categories = np.array(test_df.loc[test_df.category==g , 'category'])
    encd_categories = np.append(encd_categories, categories)
    # get the filtered headlines, encode them, and append 2d array to a list
    headlines = np.array(test_df.loc[test_df.category==g , 'headline'])
    enc_test_start = time.time()
    encs = use(headlines)
    enc_test_end = time.time()
    print(f'Encoding {len(headlines)} test phrases for {g} took {enc_test_end - enc_test_start} seconds')
    print(f'encoding rate: {len(headlines) / (enc_test_end - enc_test_start)} phrases/sec')
    encd_headlines = np.append(encd_headlines, encs['phrases'])
    encd_vects_list.append(encs['vectors'])

# this was the only way I could figure out how to compile a 2d array from the multiple encodings starting from an empty variable
encd_vects = np.concatenate(encd_vects_list, axis=0)

# classify the test headlines using the loaded model
calssification_start = time.time()
predictions = np.array([])
for i in range(len(encd_headlines)):
    # get corresponding headline and vector, and format as the output of USE() for input to classify()
    test_enc = {'phrases': [encd_headlines[i]], 'vectors': encd_vects[i]}
    pred_df = classify_use_avgs2vect(model, test_enc)
    pred_str = pred_df.head(1).index[0]
    predictions = np.append(predictions, pred_str)

calssification_end = time.time()
print(f'classification of test data took: {calssification_end - calssification_start} seconds')

# compile results df
d = {'headline': encd_headlines, 'category':encd_categories, 'prediction':predictions}
results_df = pd.DataFrame(data= d)

# calculate accuracy for each df row and add results as column
a = []
for i in range(len(results_df.headline.values)):
    if results_df['category'].iloc[i] == results_df['prediction'].iloc[i]:
        a.append('correct')
    else:
        a.append('wrong')
results_df['accuracy'] = a

accuracy_computation_end = time.time()
print(f'test calssification accuracy took: {accuracy_computation_end - calssification_end} seconds')

# print results
print(results_df['accuracy'].value_counts())
print(len(results_df))
results_counts = results_df['accuracy'].value_counts()
print(f"model testing accuracy: {results_counts['correct']/sum(results_counts)}")

testing_end = time.time()
print(f'Total testing time: {testing_end - testing_start}')



#  ======================================
# UPDATED TO USING JSON SERIALIZATION FOR SAVING TO MODEL_STORE
# save trained model to model store as json file
save_option = input("Want to save this trained model? y or n ").lower()
if save_option == 'y':
    modelJSON(model)
else:
    print("model was not saved")
