import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import time

from triage import configs
from triage.api_helper_code import Training_object, Predict_object, eval_accur


def main():

    # get and pre-process data
    # This first code block is data-set-specific

    data = pd.read_csv(configs.data_path)
    data.drop(['x1', 'x2'], axis=1, inplace=True)  # drop unneeded columns
    data.drop_duplicates(inplace=True)  # remove dupliacte phrases from dataset

    # # view data formatting
    # print(data.columns.tolist())
    # print(data['class'].value_counts())

    # # filter the data set to just relevant columns for groups of interest
    # filters = ['abdomen', 'chest', 'mouth_face', 'sexual_health']
    query = ((data['class'] == 'abdomen') | (data['class'] == 'chest') | (data['class'] == 'mouth_face') | (data['class'] == 'sexual_health'))
    filtered_data = data.loc[query,['Disease','class']]
    # print(filtered_data['class'].value_counts())

    # assign series to corresponding variables for train-test split
    x_phrases = filtered_data['Disease']  # series containing string phrases
    y_labels = filtered_data['class']  # series containing class labels

    # ===============================================
    # split orig data DF into train and test pd.series

    x_train, x_test, y_train, y_test = train_test_split(x_phrases, y_labels, test_size=configs.test_size,
                                                        random_state=configs.random_state)  # TODO: stratify
    # log data sub-sample sizes
    print(f'Total size of the filtered data set: {len(x_test) + len(x_train)}')
    print(f'number of training phrases: {len(x_train)}')
    print(f'number of test phrases: {len(x_test)}')
    print(f'proportion of held-out test data for prediction: {len(x_test)/len(filtered_data)}')  #this should equal the test_size param in train_test_split()
    print('=====' * 10)

    # ===============================================
    # /TRAIN

    # Instantiate Train_object and pass data
    model_train_obj = Training_object(x_train, y_train)

    # use Train_object.format_request() to compile request object text
    model_train_obj.format_request()
    # print(model_train_obj.request_obj)

    # make API request
    train_start_time = time.time()
    train_response = requests.post(configs.train_endpoint, json=model_train_obj.request_obj, timeout=30)
    train_end_time = time.time()
    print(f'\nTraining response status: {train_response.status_code}')  # log server response status code
    print(f'Training time: {train_end_time - train_start_time} seconds')

    # ===============================================
    # /PREDICT

    # Instantiate Train_object and pass data
    model_test_obj = Predict_object(x_test, y_test)

    # use Train_object.format_request() to compile request object text
    model_test_obj.format_request()
    # print(model_test_obj.request_obj)

    # make API request
    predict_start_time = time.time()
    predict_response = requests.post(configs.predict_endpoint, json=model_test_obj.request_obj, timeout=30)
    predict_end_time = time.time()
    predict_time_delta = predict_end_time - predict_start_time

    print(f'\nPredict response status: {predict_response.status_code}')  # log server response status code
    print(f'Predict request time: {predict_time_delta} seconds')
    # print(f'Time per record: {predict_time_delta/len(x_test)} seconds')

    # ====================================================
    # Evaluate classifier accuracy on TEST data
    eval_accur(model_test_obj.labels, predict_response.json())

    # ====================================================
    # Evaluate classifier accuracy on TRAIN data

    # Instantiate Train_object and pass data
    train_predict_obj = Predict_object(x_train, y_train)

    # use Train_object.format_request() to compile request object text
    train_predict_obj.format_request()
    # print(model_test_obj.request_obj)

    # make API request
    predict_start_time = time.time()
    train_predict_response = requests.post(configs.predict_endpoint, json=train_predict_obj.request_obj, timeout=30)
    predict_end_time = time.time()
    predict_time_delta = predict_end_time - predict_start_time

    print(f'\nPredict response status: {train_predict_response.status_code}')  # log server response status code
    print(f'Predict request time: {predict_time_delta} seconds')

    # ====================================================
    # Evaluate classifier accuracy on TRAIN data
    eval_accur(train_predict_obj.labels, train_predict_response.json())


if __name__ == '__main__':
    main()