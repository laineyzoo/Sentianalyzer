import csv
import datetime
from twitter import *

#row format: class, tweet

twitter = Twitter(auth=OAuth(consumer_key="6j6LaaBDPQml9nBEoCMh6q8tm",
                             consumer_secret="RevTuB9ZgD64sWoAeADWiLr9CWtKhjjeUE5jsxBsbv9dbkdS8T",
                             token="73861823-iBLrTp9UgM0CJ4SbwkmMKG3siwBiqNEFdBz7wzxyH",
                             token_secret="EROqbOf2TK69ROvZJow47FkOop3z14OqTGyBkdZ1tZZq3"))

## NEUTRAL DATASET
twitter_accounts = ["nytimes",
                    "guardian",
                    "guardiannews",
                    "BBCNews",
                    "AJEnglish",
                    "AP",
                    "Reuters",
                    "cnni",
                    "TechCrunch",
                    "sciam",
                    "FinancialTimes",
                    "business",
                    "Wikipedia",
                    "BBCBreaking",
                    "cnnbrk",
                    "BreakingNews"]

data = list()
for account in twitter_accounts:
    tweets = twitter.statuses.user_timeline(screen_name=account,count=200,lang="en")
    
    for t in tweets:
        tup = (2,t['text'])
        data.append(tup)

#write all tweets to csv files
filename = ("neutral_tweets_"+datetime.datetime.now().strftime("%I%m%p%B%d%Y"))
with open(filename,"w") as out:
    csv_out = csv.writer(out)
    for row in data:
        csv_out.writerow(row)


print("Neutral Data: Done writing", len(data), "rows!")

