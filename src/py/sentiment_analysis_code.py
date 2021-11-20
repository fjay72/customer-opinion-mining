#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import string
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")
reviews_df = pd.read_csv("/home/srikar/Downloads/reviews.csv")
reviews_df.head()


# In[2]:


reviews_df.info()


# In[3]:


# VADER sentiment analysis tool for getting Compound score.
def sentimental(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    return score

# VADER sentiment analysis tool for getting pos, neg and neu.
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'


# In[4]:


reviews_df['final_review_mixed'] = reviews_df[reviews_df.columns[3:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)
reviews_df["final_review_mixed"]


# In[5]:


reviews_df


# In[6]:


from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(pos_tag):
	if pos_tag.startswith('J'):
		return wordnet.ADJ
	elif pos_tag.startswith('V'):
		return wordnet.VERB
	elif pos_tag.startswith('N'):
		return wordnet.NOUN
	elif pos_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN

def clean_text(text):
	# lower text
	text = text.lower()
	# tokenize text and remove puncutation
	text = [word.strip(string.punctuation) for word in text.split(" ")]
	# remove words that contain numbers
	text = [word for word in text if not any(c.isdigit() for c in word)]
	# remove stop words
	stop = stopwords.words('english')
	text = [x for x in text if x not in stop]
	# remove empty tokens
	text = [t for t in text if len(t) > 0]
	# pos tag text
	pos_tags = pos_tag(text)
	# lemmatize text
	text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
	# remove words with only one letter
	text = [t for t in text if len(t) > 1]
	# join all
	text = " ".join(text)
	return(text)

reviews_df["review_clean"] = reviews_df["final_review_mixed"].apply(lambda x: clean_text(str(x)))


# In[7]:


reviews_df.head()


# In[9]:


reviews_df['Sentiment_Score']=reviews_df['review_clean'].apply(lambda x: sentimental_Score(x))
reviews_df.head()


# In[10]:


pos = reviews_df.loc[reviews_df['Sentiment_Score'] == 'pos']
neg = reviews_df.loc[reviews_df['Sentiment_Score'] == 'neg']
neu = reviews_df.loc[reviews_df['Sentiment_Score'] == 'neu']


# In[11]:


import numpy as np
pos_df = pos
neg_df = neg
neu_df = neu

pos_freq = pd.Series(np.concatenate([x.split() for x in pos_df.review_clean])).value_counts()
neg_freq = pd.Series(np.concatenate([x.split() for x in neg_df.review_clean])).value_counts()
neu_freq = pd.Series(np.concatenate([x.split() for x in neu_df.review_clean])).value_counts()


# In[36]:





# In[12]:


neg_string = str(neg['final_review_mixed'])
pos_string = str(pos['final_review_mixed'])
neu_string = str(neu['final_review_mixed'])

pos_data= pos_string.split()
neg_data= neg_string.split()
neu_data= neu_string.split()

