import tweepy
import requests


bearer_token = "AAAAAAAAAAAAAAAAAAAAAOIqcgEAAAAA0Hng3Wn9CWjR9mqZfF%2Ff4Ee1LgQ%3Deu8UStPJXQv5VdlyESSNkekDFAUipmTn7fgXhWQjOhgZXcwRTG";
consumer_key = "2hzhDpiN1p7l58BurF0D9K0qx"
consumer_secret = "BKw7zsx9XKeVKzC6yFBTAeJsfrldFpe832xGUnq4dAOQBXabH0"
access_token = "1270213391622328322-utEDMCzzun1a49nkHUkqhG8vqIHjMV"
access_token_secret = "TU7LALdKlJWQgLuhLb7VGXNDnf9Q5pyE0tMEvnRlPsrEi"


client = tweepy.Client( bearer_token=bearer_token,
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret,
                        return_type = requests.Response,
                        wait_on_rate_limit=True)

query = 'from:Cobratate -is:retweet'

tweets = client.search_recent_tweets(query=query,
                                    tweet_fields=['author_id', 'created_at'],
                                     max_results=100)

import pandas as pd

tweets_dict = tweets.json()
tweets_data = tweets_dict['data']
df = pd.json_normalize(tweets_data)
df.to_csv("tweets-tate.csv")
print(df.columns)
