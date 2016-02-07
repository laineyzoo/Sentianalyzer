import numpy as np
import csv
import re
import time
import sys
from collections import Counter
from sklearn.metrics import confusion_matrix
from twitter import *

#sentiment classes
negative_class = 0
neutral_class = 2
positive_class = 4

#regex patterns
patterns = {'username':'@([a-zA-Z_][a-zA-Z_0-9]*)',
               'links':'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[~!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        #'punctuation':'[~!*\(\),\?\\:-]',
         'punctuation':'[~!*,\?\"]',
         'hashtags':'#([a-zA-Z]|[0-9])+',
   'repeating_letters':'(\w{2})\1+'}

#stopwords file
file = open("stopwords.txt","r")
stopwords = file.read().strip().split("\n")
file.close()

#emoticons files
file = open("emoticons.txt","r")
emoticons = file.read().strip().split("\n")
file.close()

#global vars
vocab_len = 0
pos_vocab_len = 0
neg_vocab_len = 0
neutral_vocab_len = 0

#pos_class_prior
#neg_class_prior
#neutral_class_prior

positive_counts = Counter()
negative_counts = Counter()
neutral_counts = Counter()

def tweet_preprocessing(tweet_list):
    processed_tweets = list()

    for tweet in tweet_list:
        clean_tweet = tweet
        
        #change everything to lowercase
        clean_tweet = clean_tweet.lower()
        
        #remove @username
        pattern = re.compile(patterns['username'])
        clean_tweet = pattern.sub("",clean_tweet)

        #remove links
        pattern = re.compile(patterns['links'])
        clean_tweet = pattern.sub("",clean_tweet)
        
        #remove hashtags
        pattern = re.compile(patterns['hashtags'])
        clean_tweet = pattern.sub("",clean_tweet)

        #remove stopwords
        tweet_words = clean_tweet.split(" ")
        tweet_words = [word for word in tweet_words if word.lower() not in stopwords]
        clean_tweet = " ".join(tweet_words)
        
        #remove hashtags
        
        
        #replace characters repeated more than 2 times with 2 copies
        clean_tweet = re.sub(r'(\w{2})\1+', r'\1', clean_tweet)

        #remove emoticons
        tweet_words = clean_tweet.split(" ")
        tweet_words = [word for word in tweet_words if word.lower() not in emoticons]
        clean_tweet = " ".join(tweet_words)
        
        #remove punctuation
        pattern = re.compile(patterns['punctuation'])
        clean_tweet = pattern.sub("",clean_tweet)

        processed_tweets.append(clean_tweet)

    return processed_tweets


def train_analyzer(train_labels, train_tweets):


    ####### TRAINING #######
    
    all_labels = np.array(train_labels)
    all_tweets = np.array(train_tweets)

    pos_tweets = all_tweets[all_labels[:]==positive_class]
    neg_tweets = all_tweets[all_labels[:]==negative_class]
    neutral_tweets = all_tweets[all_labels[:]==neutral_class]

    #get the entire vocabulary into a python list
    vocab_str = " ".join(train_tweets)
    vocab_set = set(vocab_str.split(" ")) #remove duplicates
    vocabulary = list(vocab_set)

    global vocab_len #no of words in the entire corpus
    vocab_len = len(vocabulary)

    #get positive tweets vocabulary into a python list - keep duplicates
    pos_tweets_str = " ".join(pos_tweets)
    pos_tweets_vocab = pos_tweets_str.split(" ")

    global pos_vocab_len #no of words in all positive tweets
    pos_vocab_len = len(pos_tweets_vocab)

    #get negative tweets vocabulary into a python list - keep duplicates
    neg_tweets_str = " ".join(neg_tweets)
    neg_tweets_vocab = neg_tweets_str.split(" ")

    global neg_vocab_len #no of words in all negative tweets
    neg_vocab_len = len(neg_tweets_vocab)

    #get neutral tweets vocabulary into a python list - keep duplicates
    neutral_tweets_str = " ".join(neutral_tweets)
    neutral_tweets_vocab = neutral_tweets_str.split(" ")

    global neutral_vocab_len #no of words in all neutral tweets
    neutral_vocab_len = len(neg_tweets_vocab)

    #words frequnecy per class
    global positive_counts
    positive_counts = Counter(pos_tweets_vocab)

    global negative_counts
    negative_counts = Counter(neg_tweets_vocab)

    global neutral_counts
    neutral_counts = Counter(neutral_tweets_vocab)

    #priors
    #global pos_class_prior = len(pos_tweets)/len(all_tweets)
    #global neg_class_prior = len(neg_tweets)/len(all_tweets)
    #global neutral_class_prior = len(neutral_tweets)/len(all_tweets)


def test_analyzer(test_labels, test_tweets):

    predict_labels = list()
    
    #predict NB class for each tweet
    for tweet in test_tweets:
        
        words = tweet.split(" ")
        c_map = list()

        for c in range(3):

            likelihood = 0
            
            for word in words:
                if c==0:
                    likelihood += np.log((negative_counts[word]+1) / (neg_vocab_len+vocab_len))
                elif c==1:
                    likelihood += np.log((neutral_counts[word]+1) / (neutral_vocab_len+vocab_len))
                elif c==2:
                    likelihood += np.log((positive_counts[word]+1) / (pos_vocab_len+vocab_len))

            c_map.append(likelihood)

        #label tweet with class with highest prob
        label = 2*(c_map.index(max(c_map)))
        predict_labels.append(label)

    #get the confusion matrix
    conf_matrix = confusion_matrix(test_labels,predict_labels)
    accuracy = sum(np.diagonal(conf_matrix))/len(test_labels)
    
    print("Confusion Matrix:")
    print (conf_matrix)
    
    print("Actual:",np.bincount(np.array(test_labels)))
    print("Predicted:",np.bincount(np.array(predict_labels)))
    
    print("Accuracy: ", accuracy*100)



def shuffle_data(labels, tweets):

    labels_shuffle = []
    tweets_shuffle = []

    index_shuffle = np.arange(len(labels))
    np.random.shuffle(index_shuffle)

    for i in index_shuffle:
        labels_shuffle.append(labels[i])
        tweets_shuffle.append(tweets[i])

    return (labels_shuffle, tweets_shuffle)


######################### TWITTER ############################

def twitter_auth():
    twitter = Twitter(auth=OAuth(consumer_key="6j6LaaBDPQml9nBEoCMh6q8tm",
                                 consumer_secret="RevTuB9ZgD64sWoAeADWiLr9CWtKhjjeUE5jsxBsbv9dbkdS8T",
                                 token="73861823-iBLrTp9UgM0CJ4SbwkmMKG3siwBiqNEFdBz7wzxyH",
                                 token_secret="EROqbOf2TK69ROvZJow47FkOop3z14OqTGyBkdZ1tZZq3"))
    return twitter

def classify_tweets(search_hashtag):
    
    twitter = twitter_auth()
    
    #collect tweets with polling
    wait = int(sys.argv[3])#120
    repeat = int(sys.argv[2])#3
    count = 0
    tweets = list()
    tweets_orig = list()
    while count < repeat:
        reply = twitter.search.tweets(q=search_hashtag,count=200,lang="en")
        statuses = reply['statuses']
        
        for i in range(len(statuses)):
            tweets.append(statuses[i]['text'])
            tweets_orig.append(statuses[i]['text'])
        
        count+=1
        if count < repeat:
            time.sleep(wait)

    tweets = tweet_preprocessing(tweets)
    train_analyzer(train_labels, train_tweets)

    #classify the new tweets
    predict_label = list()
    tweet_with_label = list()

    
    #predict NB class for each tweet
    for t in range(len(tweets)):
        
        words = tweets[t].split(" ")
        c_map = list()
        
        for c in range(3):
            
            likelihood = 0
            
            for word in words:
                if c==0:
                    likelihood += np.log((negative_counts[word]+1) / (neg_vocab_len+vocab_len))
                elif c==1:
                    likelihood += np.log((neutral_counts[word]+1) / (neutral_vocab_len+vocab_len))
                elif c==2:
                    likelihood += np.log((positive_counts[word]+1) / (pos_vocab_len+vocab_len))

            
            c_map.append(likelihood)
        
        #assign this tweet to class with max MAP
        label = c_map.index(max(c_map))
        predict_label.append(label)
        tweet_with_label.append((label-1, tweets_orig[t]))

    #print summary of classification
    summary = np.bincount(predict_label)
    print("Classified", len(tweets), "tweets for", search_hashtag,":")
    print("Positive tweets: ",summary[2])
    print("Negative tweets: ",summary[0])
    print("Neutral tweets: ", summary[1])

    #write predicted labels to file
    filename = ("tweets_"+search_hashtag)
    with open(filename,"w") as out:
        csv_out = csv.writer(out)
        for row in tweet_with_label:
            csv_out.writerow(row)
    print("Done writing", len(tweet_with_label), "rows!")


def k_fold_validation(K, train_labels, train_tweets):
    #k-fold validation
    K = 10
    labels_folds = np.array_split(train_labels, K)
    tweets_folds = np.array_split(train_tweets, K)

    for fold in range(K):
    
        print("Fold ",fold)
    
        k_fold_labels = np.delete(labels_folds,fold,axis=0)
        k_fold_labels = k_fold_labels.flatten()
    
        k_fold_tweets = np.delete(tweets_folds,fold,axis=0)
        k_fold_tweets = k_fold_tweets.flatten()
    
        train_analyzer(k_fold_labels, k_fold_tweets)
        test_analyzer(labels_folds[fold], tweets_folds[fold])
    
        print("-----------------------------------")

########################### MAIN ##############################

trainfile = open("dataset_20k.csv","r",encoding='utf-8', errors='ignore')
reader = csv.reader(trainfile)
data = list(reader)
train_len = len(data)

train_labels = list()
train_tweets = list()

for i in range(train_len):
    train_labels.append(int(data[i][0]))
    train_tweets.append(data[i][1])

#shuffle the training data for k-fold validation
(train_labels, train_tweets) = shuffle_data(train_labels, train_tweets)

#clean-up tweets
train_tweets = tweet_preprocessing(train_tweets)

#do k-fold validation
#k=10
#k_fold_validation(k,train_labels,train_tweets)

#classify actual tweets

#get command-line args
hashtag = "#"+str(sys.argv[1])
classify_tweets(hashtag)


##twitter API code##
#from twitter import *
#twitter = Twitter(auth=OAuth(consumer_key="6j6LaaBDPQml9nBEoCMh6q8tm",
#                             consumer_secret="RevTuB9ZgD64sWoAeADWiLr9CWtKhjjeUE5jsxBsbv9dbkdS8T",
#                             token="73861823-iBLrTp9UgM0CJ4SbwkmMKG3siwBiqNEFdBz7wzxyH",
#                             token_secret="EROqbOf2TK69ROvZJow47FkOop3z14OqTGyBkdZ1tZZq3"))
#twitter.search.tweets(q="search term",count=100) #max 100
#twiiter.statuses.user_timeline(screen_name="nytimes",count=200) #max 200


