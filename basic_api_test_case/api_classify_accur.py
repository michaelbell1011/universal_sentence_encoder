
def evaluate(_y_test, _api_response_object):
    """
    Logic to evaluate and display the accuracy of Semantic Similarity Classifier API

    :param _y_test: pd.array of actual category labels for test records from split data set
    :param _api_response_object: Semantic Similarity Classifier API response JSON parsed as dict
    :return: None
    """

    print("=====" * 10)
    predictions = _api_response_object['labels']  # list of prediction labels
    print(f'Number of predicted labels: {len(predictions)}')

    actuals = list(_y_test)  # cast pd.array to list
    print(f'Number of actual labels: {len(actuals)}')

    compare = ["correct" if (prediction == actuals[i]) else "wrong" for i, prediction in enumerate(predictions)]
    accuracy = compare.count("correct")/len(compare)
    print(f'Classification accuracy for {len(predictions)} test records: {accuracy*100}%')
    print("=====" * 10)
