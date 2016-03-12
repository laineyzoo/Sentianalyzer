
# This version for deploymnet

import cgi, cgitb
import numpy as np
import csv
import re
import time
from collections import Counter
from twitter import *
from flask import Flask
from flask import request
from flask import render_template

#sentiment classes
negative_class = 0
neutral_class = 2
positive_class = 4

#regex patterns
patterns = {'username':'@([a-zA-Z_][a-zA-Z_0-9]*)([~!*\"\.\(\),\?\\:;-]*)',
               'links':'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[~!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(//)*',
              'links2':'http[s]?:(/)*',
         'punctuation':"(-)+|\||@|[|~|!|*|\"|(\.)+|\(|\)|,|\?|\\|:|;|_|[|]|\/|\'|=",
        #'punctuation':'[~!*,\?\"]',
            'hashtags':'#([a-zA-Z]|[0-9])+',
   'repeating_letters':'(\w{2})\1+'}

#stopwords file
file = open("stopwords.txt","r")
stopwords = file.read().strip().split("\n")
file.close()

#negation words file
file = open("negations.txt","r")
negations = file.read().strip().split("\n")
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
        
        pattern = re.compile(patterns['links2'])
        clean_tweet = pattern.sub("",clean_tweet)
        
        #remove hashtags
        pattern = re.compile(patterns['hashtags'])
        clean_tweet = pattern.sub("",clean_tweet)
        
        #remove stopwords
        tweet_words = clean_tweet.split()
        tweet_words = [word for word in tweet_words if word.lower() not in stopwords]
        clean_tweet = " ".join(tweet_words)
        
        
        #deal with negations --> example: didn't like episode -> didn't not_like episode
        tweet_words = clean_tweet.split()
        intersect = list(set(negations).intersection(set(tweet_words)))
        for word in intersect:
            idx = [i for i,x in enumerate(tweet_words) if x==word]
            for i in idx:
                if i+1 < len(tweet_words):
                    tweet_words[i+1] = "not_"+tweet_words[i+1]
        clean_tweet = " ".join(tweet_words)
        
        #replace characters repeated more than 2 times with 2 copies
        clean_tweet = re.sub(r'(\w{2})\1+', r'\1', clean_tweet)
        
        #remove emoticons
        tweet_words = clean_tweet.split()
        tweet_words = [word for word in tweet_words if word.lower() not in emoticons]
        clean_tweet = " ".join(tweet_words)
        
        #remove punctuation
        pattern = re.compile(patterns['punctuation'])
        clean_tweet = pattern.sub("",clean_tweet)
        
        #remove stopwords
        tweet_words = clean_tweet.split()
        tweet_words = [word for word in tweet_words if word.lower() not in stopwords]
        clean_tweet = " ".join(tweet_words)

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
    vocab_set = set(vocab_str.split()) #remove duplicates
    vocabulary = list(vocab_set)

    global vocab_len #no of words in the entire corpus
    vocab_len = len(vocabulary)

    #get positive tweets vocabulary into a python list - keep duplicates
    pos_tweets_str = " ".join(pos_tweets)
    pos_tweets_vocab = pos_tweets_str.split()

    global pos_vocab_len #no of words in all positive tweets
    pos_vocab_len = len(pos_tweets_vocab)

    #get negative tweets vocabulary into a python list - keep duplicates
    neg_tweets_str = " ".join(neg_tweets)
    neg_tweets_vocab = neg_tweets_str.split()

    global neg_vocab_len #no of words in all negative tweets
    neg_vocab_len = len(neg_tweets_vocab)

    #get neutral tweets vocabulary into a python list - keep duplicates
    neutral_tweets_str = " ".join(neutral_tweets)
    neutral_tweets_vocab = neutral_tweets_str.split()

    global neutral_vocab_len #no of words in all neutral tweets
    neutral_vocab_len = len(neg_tweets_vocab)

    #words frequnecy per class
    global positive_counts
    positive_counts = Counter(pos_tweets_vocab)

    global negative_counts
    negative_counts = Counter(neg_tweets_vocab)

    global neutral_counts
    neutral_counts = Counter(neutral_tweets_vocab)


def compute_tf_idf(labels, tweets, n):
    
    all_tweets = np.array(tweets)
    all_labels = np.array(labels)
    
    all_tweets_words = list() #1-d vector of words from all tweets minus duplicated words in a tweet
    pos_tweets_words = list() #1-d vector of words from positive tweets minus duplicated words in a tweet
    neg_tweets_words = list() #1-d vector of words from negative tweets minus duplicated words in a tweet
    newt_tweets_words = list() #1-d vector of words from neutral tweets minus duplicated words in a tweet
    
    pos_tfidf_scores = list()
    neg_tfidf_scores = list()
    newt_tfidf_scores = list()
    
    for i in range(len(tweets)):
        all_tweets_words.extend(tweets[i].split())
    
    pos_tweets = all_tweets[all_labels[:]==2]
    for i in range(len(pos_tweets)):
        pos_tweets_words.extend(pos_tweets[i].split())
    
    neg_tweets = all_tweets[all_labels[:]==0]
    for i in range(len(neg_tweets)):
        neg_tweets_words.extend(neg_tweets[i].split())
    
    newt_tweets = all_tweets[all_labels[:]==1]
    for i in range(len(newt_tweets)):
        newt_tweets_words.extend(newt_tweets[i].split())


    all_counts = Counter(all_tweets_words)
    pos_counts = Counter(pos_tweets_words)
    neg_counts = Counter(neg_tweets_words)
    newt_counts = Counter(newt_tweets_words)

    pos_words_unique = list(set(pos_tweets_words))
    neg_words_unique = list(set(neg_tweets_words))
    newt_words_unique = list(set(newt_tweets_words))
    
    
    #compute the tf-idf score of each unique word in each class
    
    for word in pos_words_unique:
        df = 1
        if neg_counts[word] > 0:
            df+=1
        if newt_counts[word] > 0:
            df+=1
        
        score = (1+np.log10(pos_counts[word]))*(np.log10(3/df))
        pos_tfidf_scores.append(score)


    for word in neg_words_unique:
        df = 1
        if pos_counts[word] > 0:
            df+=1
        if newt_counts[word] > 0:
            df+=1
        
        score = (1+np.log10(neg_counts[word]))*(np.log10(3/df))
        neg_tfidf_scores.append(score)


    for word in newt_words_unique:
        df = 1
        if pos_counts[word] > 0:
            df+=1
        if neg_counts[word] > 0:
            df+=1
        
        score = (1+np.log10(newt_counts[word]))*(np.log10(3/df))
        newt_tfidf_scores.append(score)

    #sort words for each class by its tf-idf score
    pos_words_ranked = [a for (b,a) in sorted(zip(pos_tfidf_scores,pos_words_unique))]
    neg_words_ranked = [a for (b,a) in sorted(zip(neg_tfidf_scores,neg_words_unique))]
    newt_words_ranked = [a for (b,a) in sorted(zip(newt_tfidf_scores,newt_words_unique))]
    
    #return the n-highest ranking words for each class
    return (pos_words_ranked[len(pos_words_ranked)-n:len(pos_words_ranked)],
            neg_words_ranked[len(neg_words_ranked)-n:len(neg_words_ranked)],
            newt_words_ranked[len(newt_words_ranked)-n:len(newt_words_ranked)])




def compute_mutual_information(labels, tweets, n):
    
    all_tweets = np.array(tweets)
    all_labels = np.array(labels)
    
    all_tweets_words = list() #1-d vector of words from all tweets minus duplicated words in a tweet
    pos_tweets_words = list() #1-d vector of words from positive tweets minus duplicated words in a tweet
    neg_tweets_words = list() #1-d vector of words from negative tweets minus duplicated words in a tweet
    newt_tweets_words = list() #1-d vector of words from neutral tweets minus duplicated words in a tweet
    
    pos_mi_scores = list()
    neg_mi_scores = list()
    newt_mi_scores = list()
    
    for i in range(len(tweets)):
        all_tweets_words.extend(tweets[i].split())
    
    pos_tweets = all_tweets[all_labels[:]==2]
    for i in range(len(pos_tweets)):
        pos_tweets_words.extend(pos_tweets[i].split())
    
    neg_tweets = all_tweets[all_labels[:]==0]
    for i in range(len(neg_tweets)):
        neg_tweets_words.extend(neg_tweets[i].split())
    
    newt_tweets = all_tweets[all_labels[:]==1]
    for i in range(len(newt_tweets)):
        newt_tweets_words.extend(newt_tweets[i].split())


    all_counts = Counter(all_tweets_words)
    pos_counts = Counter(pos_tweets_words)
    neg_counts = Counter(neg_tweets_words)
    newt_counts = Counter(newt_tweets_words)

    pos_words_unique = list(set(pos_tweets_words))
    neg_words_unique = list(set(neg_tweets_words))
    newt_words_unique = list(set(newt_tweets_words))
    
    for word in pos_words_unique:
        
        n11 = pos_counts[word]
        n01 = len(pos_tweets_words)-n11
        n10 = max((neg_counts[word] + pos_counts[word]),1)
        n00 = (len(neg_tweets_words)+ len(newt_tweets_words)) - n10
        ntotal = len(all_tweets_words)
        
        
        mi_score = (n11/ntotal)*np.log2((ntotal*n11)/((n11+n10)*(n11+n01)))+(n01/ntotal)*np.log2(((ntotal*n01)/((n01+n11)*(n01+n00)))+(n10/ntotal)*np.log2((ntotal*n10)/((n10+n11)*(n10+n00))))+(n00/ntotal)*np.log2((ntotal*n00)/((n00+n10)*(n00+n01)))
        pos_mi_scores.append(mi_score)


    for word in neg_words_unique:
        
        n11 = neg_counts[word]
        n01 = len(neg_tweets_words)-n11
        n10 = max((neg_counts[word] + pos_counts[word]),1)
        n00 = (len(pos_tweets_words)+ len(newt_tweets_words)) - n10
        ntotal = len(all_tweets_words)
        
        
        mi_score = (n11/ntotal)*np.log2((ntotal*n11)/((n11+n10)*(n11+n01)))+(n01/ntotal)*np.log2((ntotal*n01)/((n01+n11)*(n01+n00)))+(n10/ntotal)*np.log2((ntotal*n10)/((n10+n11)*(n10+n00)))+(n00/ntotal)*np.log2((ntotal*n00)/((n00+n10)*(n00+n01)))
        
        neg_mi_scores.append(mi_score)


    for word in newt_words_unique:
    
        n11 = newt_counts[word]
        n01 = len(newt_tweets_words)-n11
        n10 = max((neg_counts[word] + pos_counts[word]),1)
        n00 = (len(neg_tweets_words)+ len(pos_tweets_words)) - n10
        ntotal = len(all_tweets_words)
        
        
        mi_score = (n11/ntotal)*np.log2((ntotal*n11)/((n11+n10)*(n11+n01)))+(n01/ntotal)*np.log2((ntotal*n01)/((n01+n11)*(n01+n00)))+(n10/ntotal)*np.log2((ntotal*n10)/((n10+n11)*(n10+n00)))+(n00/ntotal)*np.log2((ntotal*n00)/((n00+n10)*(n00+n01)))
        
        newt_mi_scores.append(mi_score)


    #sort words for each class by its mi score
    pos_words_ranked = [a for (b,a) in sorted(zip(pos_mi_scores,pos_words_unique))]
    neg_words_ranked = [a for (b,a) in sorted(zip(neg_mi_scores,neg_words_unique))]
    newt_words_ranked = [a for (b,a) in sorted(zip(newt_mi_scores,newt_words_unique))]

    #return the n-highest ranking words for each class
    return (pos_words_ranked[len(pos_words_ranked)-n:len(pos_words_ranked)],
            neg_words_ranked[len(neg_words_ranked)-n:len(neg_words_ranked)],
            newt_words_ranked[len(newt_words_ranked)-n:len(newt_words_ranked)])


######################### TWITTER ############################

def twitter_auth():
    twitter = Twitter(auth=OAuth(consumer_key="",
                                 consumer_secret="",
                                 token="",
                                 token_secret=""))
    return twitter

def classify_tweets(search_hashtag, train_labels, train_tweets, num_queries):

    twitter = twitter_auth()
    #collect tweets with polling
    wait = 10
    repeat = num_queries
    count = 0
    tweets = list()
    tweets_orig = list()
    while count < repeat:
        reply = twitter.search.tweets(q=search_hashtag,count=200,lang="en")
        statuses = reply['statuses']
        
        for i in range(len(statuses)):
            tweets.append(statuses[i]['text'])
        
        count+=1
        if count < repeat:
            time.sleep(wait)

    tweets = tweet_preprocessing(tweets)
    train_analyzer(train_labels, train_tweets)

    #classify the new tweets
    predict_label = list()

    #predict NB class for each tweet
    for t in range(len(tweets)):
        
        words = tweets[t].split()
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

    #print summary of classification and Rotten Tomatoes data
    summary = np.bincount(predict_label)
    (rt_score, rt_says) = get_rottentomatoes_info(search_hashtag)
    (top_pos_words_mi, top_neg_words_mi, top_newt_words_mi) = compute_mutual_information(predict_label,tweets,10)
    (top_pos_words_tfidf, top_neg_words_tfidf, top_newt_words_tfidf) = compute_tf_idf(predict_label,tweets,10)
    
    return (summary, rt_score, rt_says, top_pos_words_mi, top_neg_words_mi, top_newt_words_mi, top_pos_words_tfidf, top_neg_words_tfidf, top_newt_words_tfidf)


def get_rottentomatoes_info(search_hashtag):
    
    twitter = twitter_auth()
    rt_account = "RottenTomatoes"
    rt_ratings = ["Rotten","Fresh","Certified"]
    rt_tweets = twitter.statuses.user_timeline(screen_name=rt_account,count=200,lang="en")
    
    found_hashtag = "False"
    rt_percent = ""
    rt_says = ""
    for t in rt_tweets:
        words = t['text'].split()
        
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
                rt_percent = match.group(0)
            else:
                rt_percent = "Not Found"

    return (rt_percent, rt_says)


########################### MAIN ##############################

app = Flask(__name__)

@app.route('/oscars')
def oscars():

    csvfile = open("tweets_Oscars.csv")
    reader = csv.reader(csvfile)
    data = list(reader)

    total = len(data)
    oscars_tweets = list()
    oscars_labels = list()

    for i in range(total):
        oscars_labels.append(int(data[i][0])+1)
        oscars_tweets.append(data[i][1])
            
    #clean-up tweets
    oscars_tweets = tweet_preprocessing(oscars_tweets)
    
    counts = np.bincount(np.array(oscars_labels))
    neg_count = counts[0]
    newt_count = counts[1]
    pos_count = counts[2]
    (pos_terms_mi, neg_terms_mi, newt_terms_mi) = compute_mutual_information(oscars_labels,oscars_tweets,10)
    (pos_terms_tfidf, neg_terms_tfidf, newt_terms_tfidf) = compute_tf_idf(oscars_labels,oscars_tweets,10)

    return render_template("result.html", search_term="Oscars", total_tweets=total, positive=pos_count, negative=neg_count, neutral=newt_count, rt_score="No RT data", rt_rate="", positive_terms_mi=pos_terms_mi, negative_terms_mi=neg_terms_mi, neutral_terms_mi=newt_terms_mi, positive_terms_tfidf=pos_terms_tfidf, negative_terms_tfidf=neg_terms_tfidf,neutral_terms_tfidf=newt_terms_tfidf)

@app.route('/grammys')
def grammys():
    
    csvfile = open("tweets_Grammys.csv")
    reader = csv.reader(csvfile)
    data = list(reader)
    
    total = len(data)
    tweets = list()
    labels = list()
    
    for i in range(total):
        labels.append(int(data[i][0])+1)
        tweets.append(data[i][1])
    
    #clean-up tweets
    tweets = tweet_preprocessing(tweets)
    
    counts = np.bincount(np.array(labels))
    neg_count = counts[0]
    newt_count = counts[1]
    pos_count = counts[2]
    (pos_terms_mi, neg_terms_mi, newt_terms_mi) = compute_mutual_information(labels,tweets,10)
    (pos_terms_tfidf, neg_terms_tfidf, newt_terms_tfidf) = compute_tf_idf(labels,tweets,10)

    return render_template("result.html", search_term="#Grammys", total_tweets=total, positive=pos_count, negative=neg_count, neutral=newt_count, rt_score="No RT data", rt_rate="", positive_terms_mi=pos_terms_mi, negative_terms_mi=neg_terms_mi, neutral_terms_mi=newt_terms_mi, positive_terms_tfidf=pos_terms_tfidf, negative_terms_tfidf=neg_terms_tfidf,neutral_terms_tfidf=newt_terms_tfidf)

@app.route('/')
def home():
    print("Render template")
    return render_template("home.html")

@app.route('/', methods=['POST'])
def home_post():
    
    print("Received POST")
    search_term = request.form['hashtag']
    print("hashtag: ", search_term)
    num_queries = int(request.form['num_query'])
    print("queries: ", num_queries)

    trainfile = open("dataset_20k.csv","r",encoding='utf-8', errors='ignore')
    reader = csv.reader(trainfile)
    data = list(reader)
    train_len = len(data)
    
    train_labels = list()
    train_tweets = list()
    
    for i in range(train_len):
        train_labels.append(int(data[i][0]))
        train_tweets.append(data[i][1])


    #clean-up tweets
    train_tweets = tweet_preprocessing(train_tweets)
    
    
    #classify actual tweets
    twitter = twitter_auth()
    #get command-line args
    hashtag = search_term
    (result, rt_score, rt_rate, pos_terms_mi, neg_terms_mi, newt_terms_mi, pos_terms_tfidf, neg_terms_tfidf, newt_terms_tfidf) = classify_tweets(hashtag, train_labels, train_tweets, num_queries)
    pos_count = result[2]
    neg_count = result[0]
    newt_count = result[1]
    total = sum(result)
    title = search_term.upper()

    if rt_score == "":
        rt_score = "No RT data for this topic"

    return render_template("result.html", search_term=hashtag, total_tweets=total, positive=pos_count, negative=neg_count, neutral=newt_count, rt_score=rt_score, rt_rate=rt_rate, positive_terms_mi=pos_terms_mi, negative_terms_mi=neg_terms_mi, neutral_terms_mi=newt_terms_mi, positive_terms_tfidf=pos_terms_tfidf, negative_terms_tfidf=neg_terms_tfidf,neutral_terms_tfidf=newt_terms_tfidf)
    #return 'Classified %d positive tweets, %d negative tweets and %d neutral tweets for %s.' % (positive, negative, neutral, title)


if __name__ == '__main__':
    app.run(debug=True)


