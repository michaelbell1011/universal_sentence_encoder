
import requests
import time

# custom modules
import testing_configs
import api_classify_accur
import sample_data


# # helper functions for syntax formatting of API request objects moved to api_helper_code.py module.


def api_test():
    """
    Demonstrate basic dialogue to and from sentence-similarity-classifier API for testing.
    """

    # make /train request using the correctly formatted train request object from sample_data module
    train_start_time = time.time()
    train_response = requests.post(testing_configs.train_endpoint, json=sample_data.train_obj, timeout=5)
    train_end_time = time.time()

    # log response output
    print(f'\nTraining response status: {train_response.status_code}')  # log server response status code
    print(f'Training time: {train_end_time - train_start_time} seconds')

    # # ====================================================

    # make /predict request using the correctly formatted predict request object from sample_data module
    predict_start_time = time.time()
    predict_response = requests.post(testing_configs.predict_endpoint, json=sample_data.predict_obj, timeout=5)
    predict_end_time = time.time()

    # log response output
    predict_time_delta = predict_end_time - predict_start_time
    print(f'\nPredict response status: {predict_response.status_code}')  # log server response status code
    print(f'Predict request time: {predict_time_delta} seconds')
    # print(f'Time per record: {predict_time_delta/len(x_test)} seconds')


    # # ====================================================
    # # Evaluate classifier accuracy and log output
    api_classify_accur.evaluate(sample_data.labels, predict_response.json())


if __name__ == '__main__':
    api_test()
