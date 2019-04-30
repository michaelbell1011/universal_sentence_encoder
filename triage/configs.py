data_path = "data/phrases_embed.csv"

# train-test-split parameters
test_size=0.30
random_state=30 # may opt to not set the random state for subsampling randomness


# Docker container running on localhost port 4000
train_endpoint = 'http://0.0.0.0:80/train'
predict_endpoint = "http://0.0.0.0:80/predict"