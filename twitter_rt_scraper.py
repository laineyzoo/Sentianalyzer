import re
import sys
import time
from twitter import *

#grab and parse Rotten Tomatoes tweets

def get_tweets():
    tweets = twitter.statuses.user_timeline(screen_name=rt_account,count=200,lang="en")
    return tweets

def parse_tweets(tweets,retry):
    
    found_hashtag = False
    for t in tweets:

        words = t['text'].split(" ")
        if search_hashtag in words:
            #hashtag found, start parsing tweet
            found_hashtag = True
            #search for the words Fresh, Rotten or Certified
            inter = list(set(words) & set(rt_ratings))
            if len(inter) > 0:
                rt_says = inter[0]
            else:
                rt_says = "Not Found"

            #search the percentage
            match = re.search('[0-9]+%',t['text'])
            if match is not None:
                percentage = match.group(0)
            else:
                percentage = "Not Found"

    if found_hashtag == True:
        print("Rotten Tomatoes says: "+search_hashtag+" is "+rt_says+" at "+percentage)
    else:
        if retry == False:
            #hashtag not in RT tweets, wait 60 sec and retry
            print("RT data not found. Retrying in 60 sec.")
            time.sleep(60)
            new_tweets = get_tweets()
            parse_tweets(new_tweets,True)
        else:
            print("Done retrying. Still no RT data.")


search_hashtag = "#"+str(sys.argv[1])

rt_account = "RottenTomatoes"
rt_ratings = ["Rotten","Fresh","Certified"]

twitter = Twitter(auth=OAuth(consumer_key="6j6LaaBDPQml9nBEoCMh6q8tm",
                             consumer_secret="RevTuB9ZgD64sWoAeADWiLr9CWtKhjjeUE5jsxBsbv9dbkdS8T",
                             token="73861823-iBLrTp9UgM0CJ4SbwkmMKG3siwBiqNEFdBz7wzxyH",
                             token_secret="EROqbOf2TK69ROvZJow47FkOop3z14OqTGyBkdZ1tZZq3"))



data = list()
tweets = get_tweets()
rt_says = ""
percentage = ""
parse_tweets(tweets,False)