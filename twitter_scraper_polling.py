import time
import csv
import datetime
from twitter import *

#row format: class, tweet


##POSITIVE DATASET

def get_positive_tweets(count):

    happy_tweets = twitter.search.tweets(q=":)",count=100,lang="en")
    tweets = happy_tweets['statuses']

    data = list()
    for t in tweets:
        tup = (4,t['text'])
        data.append(tup)

    #write all tweets to csv files
    filename = ("positive_tweets_"+datetime.datetime.now().strftime("%I%m%p%B%d%Y")+"_"+str(count))
    with open(filename,"w") as out:
        csv_out = csv.writer(out)
        for row in data:
            csv_out.writerow(row)


    print(str(count)+" Positive Data: Done writing", len(data), "rows!")


##NEGATIVE DATASET

def get_negative_tweets(count):
    sad_tweets = twitter.search.tweets(q=":(",count=100,lang="en")
    tweets = sad_tweets['statuses']

    data = list()
    for t in tweets:
        tup = (0,t['text'])
        data.append(tup)

    #write all tweets to csv files
    filename = ("negative_tweets_"+datetime.datetime.now().strftime("%I%m%p%B%d%Y")+"_"+str(count))
    with open(filename,"w") as out:
        csv_out = csv.writer(out)
        for row in data:
            csv_out.writerow(row)

    print(str(count)+" Negative Data: Done writing", len(data), "rows!")


#authenticate with Twitter first
twitter = Twitter(auth=OAuth(consumer_key="6j6LaaBDPQml9nBEoCMh6q8tm",
                             consumer_secret="RevTuB9ZgD64sWoAeADWiLr9CWtKhjjeUE5jsxBsbv9dbkdS8T",
                             token="73861823-iBLrTp9UgM0CJ4SbwkmMKG3siwBiqNEFdBz7wzxyH",
                             token_secret="EROqbOf2TK69ROvZJow47FkOop3z14OqTGyBkdZ1tZZq3"))

count = 81
max = 100
wait_time = 300 #in seconds
while count <= max:
    get_positive_tweets(count)
    get_negative_tweets(count)
    count+=1
    if count <= max:
        time.sleep(wait_time)