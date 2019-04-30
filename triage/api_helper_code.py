
class Training_object:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.request_obj = {}

    def format_request(self):
        self.request_obj["req_id"]= "<POST /TRAIN REQUEST ID>"
        self.request_obj["records"] = [{"record": self.data.iloc[i], "label": self.labels.iloc[i]} for i, s in
                              enumerate(self.data)]
        return self.request_obj


class Predict_object:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.request_obj = {}

    def format_request(self):
        self.request_obj["req_id"]= "<POST /TRAIN REQUEST ID>"
        self.request_obj["records"] = [{"record": i} for i in self.data]
        return self.request_obj


def eval_accur(_y_test, _api_response_object):
    """
    Logic to evaluate and display the accuracy of Semantic Similarity Classifier API

    :param _y_test: pd.array of actual category labels for test records from split data set
    :param _api_response_object: Semantic Similarity Classifier API response JSON parsed as dict
    :return: None
    """

    print("=====" * 10)
    print("CLASSFICATION ACCURACY")
    predictions = _api_response_object['labels']  # list of prediction labels
    print(f'number of predicted labels: {len(predictions)}')

    actuals = list(_y_test)  # cast pd.array to list
    print(f'number of actual labels: {len(actuals)}')

    compare = ["correct" if (prediction == actuals[i]) else "wrong" for i, prediction in enumerate(predictions)]
    print(f'number of correct classifications: {compare.count("correct")}')

    accuracy = compare.count("correct")/len(compare)
    print(f'Classification accuracy for {len(predictions)} test records: {accuracy*100}%')
    print("=====" * 10)
