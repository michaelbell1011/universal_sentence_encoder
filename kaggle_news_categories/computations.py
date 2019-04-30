import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

def Hello():
    """Says 'Hello' to The World."""
    print("Hello world!")


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


# function to evaluate the classification reliability for all phrases in a list of averaged phrase groups
# pass in list of use_avg dicts
def eval_classifer(l_o_use_avg_dicts):
    # compile one df using df.append() from list of avg dicts
    # instantiate empty df with columns 'averaged_phrases' and 'phrase_vects'
    main_df = pd.DataFrame(columns=['phrase', 'vector', 'group_name'])

    # update l_o_avgs below
    for d in l_o_use_avg_dicts:
        df_data = dict(phrase=d['averaged_phrases'], vector=list(d['phrase_vects']))
        df = pd.DataFrame(data=df_data)
        df['group_name'] = d['group_name']
        main_df = main_df.append(df, ignore_index=True)

    # make classification predictions and add to main_df
    pred_group_list = []
    for i in range(len(main_df)):
        test_enc = {'phrases': [main_df['phrase'][i]], 'vectors': main_df['vector'][i]}
        # update l_o_avgs below
        pred_df = classify_use_avgs2vect(l_o_use_avg_dicts, test_enc)
        pred_str = pred_df.head(1).index.base[0]
        pred_group_list.append(pred_str)
    main_df['pred_group'] = pred_group_list

    # calculate accuracy and add to main_df
    a = []
    for i in range(len(main_df['group_name'])):
        if main_df['group_name'].iloc[i] == main_df['pred_group'].iloc[i]:
            a.append('correct')
        else:
            a.append('wrong')
    main_df['accuracy'] = a

    print(main_df['accuracy'].value_counts())
    print(len(main_df))
    return main_df