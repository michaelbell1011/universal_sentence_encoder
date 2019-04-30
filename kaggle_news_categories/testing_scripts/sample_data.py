
# spoof json data for testing API-- format for /train request
train_obj = {"req_id": "TestTrain1234",
          "records": [
              {"record": "How do I wire money?", "label": "wire_transfer"},
              {"record": "I wanna send money to China", "label": "wire_transfer"},
              {"record": "I have lost my credit card", "label": "cards"},
              {"record": "I forgot my card at a bar", "label": "cards"},
              {"record": "How much money do I have in my account?", "label": "account_balance"},
              {"record": "What's my account balance?", "label": "account_balance"}]}

# spoof json data for testing API-- format for /predict request
predict_obj = {"req_id": "TestPredict1234",
               "records": [
                   {"record": "How do I remit money to Japan?"},
                   {"record": "I'd like to report a lost credit card"},
                   {"record": "My credit card was stolen!"},
                   {"record": "I'd like to send money to my brother"},
                   {"record": "How much cash do I have in my bank?"}]}

# print(f'\nresponse from server: {predict_response.text}')
# Sample /predict Response:
#    {
#     "labels": [
#         "wire_transfer",
#         "cards",
#         "cards",
#         "wire_transfer",
#         "account_balance"
#     ],
#     "req_id": "TestPredict1234"
#   }

labels = ["wire_transfer",
          "cards",
          "cards",
          "wire_transfer",
          "account_balance"]