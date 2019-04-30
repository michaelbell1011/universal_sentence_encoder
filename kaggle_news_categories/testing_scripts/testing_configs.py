data_path="../data/News_Category_Dataset_v2.json" #kaggle news categories csv location

# train-test-split parameters
test_size=0.30
random_state=30 # may opt to not set the random state for subsampling randomness


# various server options for testing

# train_endpoint = 'http://127.0.0.1:8000/train' #localhost:  gunicorn -w1 -b 127.0.0.1:8000 rest_api.api:app
# predict_endpoint = "http://127.0.0.1:8000/predict"

# train_endpoint = 'https://httpbin.org/post' #online HTTP request testing service https://httpbin.org
# predict_endpoint = 'https://httpbin.org/post' #online HTTP request testing service https://httpbin.org


#
# train_endpoint = 'http://18.219.233.115:8000/train' #AWS:  gunicorn -w 1 -b 0.0.0.0:8000 rest_api.api:app
# predict_endpoint = 'http://18.219.233.115:8000/predict'



# 18.222.8.106

# train_endpoint = 'http://18.222.8.106:8000/train' #AWS:  gunicorn -w 1 -b 0.0.0.0:8000 rest_api.api:app
# predict_endpoint = 'http://18.222.8.106:8000/predict'



#  18.224.63.172

# train_endpoint = 'http://18.224.63.172:8000/train' #AWS:  gunicorn -w 1 -b 0.0.0.0:8000 rest_api.api:app
# predict_endpoint = 'http://18.224.63.172:8000/predict'

# Docker container running on localhost port 4000
train_endpoint = 'http://0.0.0.0:80/train'
predict_endpoint = "http://0.0.0.0:80/predict"