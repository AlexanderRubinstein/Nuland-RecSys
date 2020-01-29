import nltk
import numpy as np
import pandas as pd
import pysparnn.cluster_index as ci
import re
import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


nltk.download('wordnet')
nltk.download('punkt')
word_net_lemmatizer = WordNetLemmatizer()

def pre_process(text):
    # lowercase
    text = text.lower()
    
    #remove tags
    text = re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ",text)
    
    text_words = nltk.word_tokenize(text)
    lemmatized_text = ""
    for word in text_words:
        lemma = word_net_lemmatizer.lemmatize(word)
        lemmatized_text += " " + lemma
    text = lemmatized_text[1:] 
    
    return text


class TasksIndex():
    def __init__(self, df, threshold=0.9):
        self.df = df
        self.threshold = threshold
        
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        self.cv = CountVectorizer(max_df=0.85, stop_words=stop_words, max_features=10000)
        docs = self.df['normalized'].tolist()
        word_count_vector = self.cv.fit_transform(docs)
    
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf_transformer.fit(word_count_vector)
        features_vec = self.tfidf_transformer.transform(word_count_vector)
        
        self.index = ci.MultiClusterIndex(features_vec, np.arange(len(docs)))
        
    def get_similar_tasks(self, message, k_tasks=3):
        message = pre_process(message)
        ftrs_vec = self.tfidf_transformer.transform(self.cv.transform([message]))
        if ftrs_vec[0].sum() == 0:
            return []
        
        tasks = self.index.search(ftrs_vec, k=k_tasks, k_clusters=2, return_distance=True)[0]
        tasks = [int(t[1]) for t in tasks if t[0] < self.threshold]    
        tasks = self.df.iloc[tasks]['text'].tolist()
        
        return tasks
    

if __name__ == "__main__":
    message = sys.argv[1]
    
    tasks_path = 'data/photo_youdo.json'
    df = pd.read_json(tasks_path, lines=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns={'body': 'text'}, inplace=True)
    df['normalized'] = df['text'].apply(lambda x: pre_process(x))
    
    index = TasksIndex(df)
    tasks = index.get_similar_tasks(message)
    if len(tasks) == 0:
        print('Hm... seems there are no relevant tasks right now, maybe try something else?')
    else:
        print('Maybe you are interested in these suggestions:')
        print()
        for t in tasks: 
            print('- ' + t)
            print()