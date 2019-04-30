import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import time

# custom modules
import testing_configs
import api_classify_accur


# helper functions
def compile_training_request_obj(x_train_series, y_train_series):
    """
    Formats data into json object for /train request

    TODO:
    mechanism for setting request ID

    :param x_train_series:
    :param y_train_series:
    :return: foramtted request object for /train
    """

    req_obj = {"req_id": "<POST /TRAIN REQUEST ID>"}
    req_obj["records"] = [{"record": x_train_series.iloc[i], "label": y_train_series.iloc[i]} for i,s in enumerate(x_train_series)]
    return req_obj

def compile_predict_request_obj(x_test_series):
    """
    Formats data into json object for /predict request

    TODO:
    mechanism for setting request ID

    :param x_test_series:
    :return: foramtted request object for /predict
    """

    req_obj = {"req_id": "<POST /PREDICT REQUEST ID>"}
    req_obj["records"] = [{"record": test_phrase} for test_phrase in x_test_series]
    return req_obj


def api_test():
    """
    Pre-processes data from CSV, then sequentially makes /train and /predict requests, and prints /predict accuracy

    :return:
    """

    # pre-process data
    # import data from CSV and filter to what is needed
    data = pd.read_json(path_or_buf= testing_configs.data_path, lines=True)
    data.drop_duplicates(subset='headline', inplace=True)  # remove duplicated headlines

    # # testing with all data-- do not apply the filters block below
    # filtered_data = data.loc[:,['category','headline']]
    # print(f'Total size of UNfiltered data set: {len(x)}')

    # filter the data set to just relevant columns for groups of interest
    filters = ((data.category == 'POLITICS') | (data.category == 'ENTERTAINMENT') | (data.category == 'WELLNESS'))  # biggest 3 categories
    filtered_data = data.loc[filters,['category','headline']]

    # split orig data DF into train and test pd.series
    x= filtered_data['headline']
    y= filtered_data['category']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testing_configs.test_size,
                                                        random_state=testing_configs.random_state, stratify=y)
    # log data sub-sample sizes
    print(f'Total size of the filtered data set: {len(x_test) + len(x_train)}')
    print(f'number of training phrases: {len(x_train)}')
    print(f'number of test phrases: {len(x_test)}')
    print(f'proportion of held-out test data for prediction: {len(x_test)/len(filtered_data)}') #this should equal the test_size param in train_test_split()

    # ====================================================
    # compile training request json object and send request
    training_object = compile_training_request_obj(x_train, y_train)
    # print(f'training object: {training_object.keys()}')
    # print(f'training object: {training_object["records"][0:5]}')

    # make /train request
    train_start_time = time.time()
    train_response = requests.post(testing_configs.train_endpoint, json=training_object, timeout=60)
    train_end_time = time.time()

    print(f'\nTraining response status: {train_response.status_code}')  # log server response status code
    print(f'Training time: {train_end_time - train_start_time} seconds')



    # # ====================================================
    # # compile predict request json object and send request
    # predict_object = compile_predict_request_obj(x_test)
    # # print(f'predict object: {predict_object.keys()}')
    # # print(f'predict object: {predict_object["records"][0:5]}')
    #
    #
    # # make /predict request
    # predict_start_time = time.time()
    # predict_response = requests.post(testing_configs.predict_endpoint, json= predict_object, timeout=60)
    # predict_end_time = time.time()
    #
    # predict_time_delta = predict_end_time - predict_start_time
    #
    # print(f'\nPredict response status: {predict_response.status_code}')  # log server response status code
    # print(f'Predict request time: {predict_time_delta} seconds')
    # print(f'Time per record: {predict_time_delta/len(x_test)} seconds')
    #
    #
    # # ====================================================
    # # Evaluate classifier accuracy for TEST data
    #
    # api_classify_accur.evaluate(y_test, predict_response.json())
    #
    # # ====================================================
    # # ====================================================
    # # Evaluate classifier accuracy for TRAIN data
    #
    # # compile predict request json object from TRAIN DATA and send request
    # x_train_predict_object = compile_predict_request_obj(x_train)
    #
    # # make /predict request
    # x_train_predict_response = requests.post(testing_configs.predict_endpoint, json= x_train_predict_object, timeout=30)
    #
    # print(f'\nPredict on training data response status: {x_train_predict_response.status_code}')
    #
    # # Evaluate classifier accuracy on training data
    # api_classify_accur.evaluate(y_train, x_train_predict_response.json())
    # # ====================================================

if __name__ == '__main__':
    api_test()
