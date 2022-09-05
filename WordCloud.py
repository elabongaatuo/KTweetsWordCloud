# setting up the environment
from nltk.tokenize import WordPunctTokenizer
import re
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from PIL import Image
import os
directory = "path/to/your/files"
wordpunct_tokenize = WordPunctTokenizer().tokenize
stopwords = set(STOPWORDS)
stopwords.update(['RutoThe5th', 'RailaThe5th'])


# declaring the twitter api authentication details
consumer_key = "CONSUMER KEY"
consumer_secret = "CONSUMER SECRET"
access_token = "ACCESS TOKEN"
access_token_secret = "ACCESS TOKEN SECRET"

# initializing the twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# declaring some variables
keyword = ['RutoThe5th', 'RailaThe5th']
limit = 500

# initiating api request for search tweets
data = []
df = pd.DataFrame(data)

for i in keyword:
    tweets = tweepy.Cursor(api.search_tweets,
                           q=i, count=100,  tweet_mode='extended').items(limit)
    column = []
    for tweet in tweets:
        column.append([tweet.full_text])
    df['{}'.format(i)] = column


print(df.head())


# declaring regex pattersn variables

emoji_pattern = re.compile(pattern="["
                           u"\U0001F600 - \U0001F64F"  # emoticons
                           u"\U0001F300 - \U0001F5fF"  # symbols&pictographs
                           u"\U0001F680 - \U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0 - \U0001F1FF"  # flags(iOs)
                           "] +", flags=re.UNICODE)

url_pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
mentions_pattern = ['@[A-Za-z0-9]+', '#[A-Za-z0-9]+']
combined_re = re.compile('|'.join(mentions_pattern))
punctuation_pattern = re.compile(r'[^\w\s]')
english_check = re.compile('[^A-Za-z0-9]+')
mobile_number_check = re.compile(r'(?:\+\d{3})?\d{3,4}\D?\d{3}\D?\d{3}')


# creating a function that will clean the text in the dataframe

def cleaning_tweets(t):
    '''
    This function cleans the tweets by removing emojis,urls, hashtags, mentions and mobile numbers
    '''
    del_emoji_pattern = re.sub(emoji_pattern, ' ', t)
    del_url_pattern = re.sub(url_pattern, ' ', del_emoji_pattern)
    del_mention = re.sub(combined_re, ' ', del_url_pattern)
    del_non_english = re.sub(english_check, ' ', del_mention)
    del_mobnumbers = re.sub(mobile_number_check, ' ', del_non_english)
    to_lower_case = del_mobnumbers.lower()
    words = wordpunct_tokenize(to_lower_case)
    result = [x for x in words if len(x) > 2]
    return (" ".join(result)).strip()


df_clean = df.applymap(lambda x: cleaning_tweets(str(x)))
print(df_clean.head(10))

# Reading Image Files
m = 1

for col in list(df_clean):
    for filename in os.listdir(directory):
        if filename.startswith(col) and filename.endswith('.png'):
            mask = np.array(Image.open(filename))
            colors = ImageColorGenerator(mask)
    # Visualizing the wordcloud
    wordcloud = WordCloud(width=1600, stopwords=stopwords, height=800, max_font_size=200,
                          collocations=False, mask=mask, color_func=colors, background_color='white', contour_color='black', contour_width=0.5).generate(str(df_clean[col]))
    f = plt.figure(figsize=(50, 50))
    f.add_subplot(1, 2, m)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('{}'.format(col), size=40)
    plt.axis('off')
    f.add_subplot(1, 2, m+1)
    plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.title('Original Image', size=40)
    plt.axis('off')

plt.show()
