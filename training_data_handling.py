import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
import gc
import os
import pickle
gc.collect()

from nltk.corpus import wordnet


def remove_hrml_urls(data_frame):
    """removes html/url links"""
    data_frame['review']=data_frame['review'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))
    return data_frame

def convert_to_lowercase(data_frame):
    """The first pre-processing step which we will do is transform our tweets 
    into lower case. This avoids having multiple copies of the same words. For example, while calculating the word count
    ,‘Analytics’ and ‘analytics’ will be taken as different words."""
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data_frame

def removing_punctuation(data_frame):
    """The next step is to remove punctuation, as it doesn’t add any extra information while 
    treating text data. Therefore removing all instances of 
    it will help us reduce the size of the training data."""
    data_frame['review']  =data_frame['review'] .str.replace('[^\w\s]','')
    return data_frame
    
def removing_stop_words(data_frame):
    """stop words (or commonly occurring words) should be removed from the text data. 
    For this purpose, we can either create a 
    list of stopwords ourselves or we can use predefined libraries."""
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    data_frame['review']=  data_frame['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return data_frame
    
def spell_correction(data_frame):
    data_frame['review'].apply(lambda x: str(TextBlob(x).correct()))
    return data_frame

def stemming(data_frame):
    """Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. 
    For this purpose, we will use PorterStemmer from the NLTK library."""
    from nltk.stem import PorterStemmer
    st = PorterStemmer()
    data_frame['review']=data_frame['review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return data_frame

def lemmatization(data_frame):
    """Lemmatization is a more effective option than stemming because it converts 
    the word into its root word, rather than just stripping the suffices. 
    It makes use of the vocabulary and does a morphological analysis to obtain the root word. 
    Therefore, we usually prefer using lemmatization over stemming."""   
    from textblob import Word
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data_frame

def remove_rare_words(data_frame,data):
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join(x for x in x.split() if x not in data))
    return data_frame

def is_english_word(word):
    chk=wordnet.synsets(word)
    if len(chk) ==0:
        return ""
    else:
        return word

def chk_english_word(data_frame):
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join([is_english_word(word) for word in x.split()]))
    return data_frame


def remove_special_chars(data_frame):
    """removes special characters"""
    data_frame['review']=data_frame['review'].str.replace('[^A-Za-z\s]+', '')
    return data_frame

def no_of_words(data_frame):
    word_count =data_frame['review'].apply(lambda x: len(str(x).split(" ")))
    return word_count
    
def no_of_chars(data_frame):
    char_count = data_frame['review'].str.len()
    return char_count

def avg_word(sentence):
  words = sentence.split()
  if len(words) <1:
      words.append('1')
  return (sum(len(word) for word in words)/len(words))

def avg_word_length(data_frame):
    avg_word_length = data_frame['review'].apply(lambda x: avg_word(x))
    return avg_word_length

def no_of_stop_words(data_frame):
    stop = stopwords.words('english')
    stop_words = data_frame['review'].apply(lambda x: len([x for x in x.split() if x in stop]))
    return stop_words

def no_of_Uppercase_words(data_frame):
    """Anger or rage is quite often expressed by writing in UPPERCASE words
    which makes this a necessary operation to identify those words."""
    upper_words = data_frame['review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    return upper_words
  


def remove_blanks(data_frame):
    data_frame['review'].replace('  ', np.nan, inplace=True)
    #data_frame['review'].replace(' ', np.nan, inplace=True)
    data_frame= data_frame.dropna(subset=['review'])
    return data_frame

def bow_count_vector(data_frame):
    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1, 2))
    X = cv.fit_transform(data_frame['review']).toarray()    
    y = data_frame.iloc[:, 1].values    
    return X,y,cv    

def tf_idf_vecttorizer(data_frame):
    #tf-idf vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,2))
    X= tfidf.fit_transform(data_frame['review']).toarray()
    y = data_frame.iloc[:, 1].values
    return X,y

def K_folds(dataX,dataY,clf,pklname):     
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3)
    predY = np.zeros((len(dataX),))
    i=1
    for train_index, test_index in kf.split(dataX):        
        X_train, X_test = dataX[train_index], dataX[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]        
        clf.fit(X_train,y_train)        
        predY[test_index]=clf.predict(X_test)        
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test,clf.predict(X_test)))
        fullpklname=""
        fullpklname=pklname+str(i)+".pkl"
        filepath=os.path.join("D:\Python\sentiment_analysis\pkl_files",fullpklname)
        save_to_pikl(clf,filepath)
        i=i+1
    return predY,clf

def save_to_pikl(data,name):
    with open(name, 'wb') as fid: 
        pickle.dump(data,fid)
    fid.close()

def load_from_pikl(name):   
    with open(name, 'rb') as fid:        
        data=pickle.load(fid)
    fid.close
    return data


df = pd.read_csv(r"D:\Python\sentiment_analysis\trainingdata\Final Data.csv", sep=',', encoding='unicode_escape')

upper_words=no_of_Uppercase_words(df)
df['upper_case_words']=upper_words

# =============================================================================
# stop_words=no_of_stop_words(df)
# df['stop_words']=stop_words
# 
# word_count=no_of_words(df)
# df['word_count']=word_count
# 
# char_count=no_of_chars(df)
# df['char_count']=char_count
# 
# avg_word_lengths=avg_word_length(df)
# df['avg_word_lengths']=avg_word_lengths
# =============================================================================



df=remove_blanks(df)
df=remove_hrml_urls(df)
df=convert_to_lowercase(df)
df=removing_punctuation(df)
df=lemmatization(df)
data_rare= pd.Series(' '.join(df['review']).split()).value_counts()[-10100:]
df=remove_rare_words(df,data_rare)

#df=chk_english_word(df)
#df=remove_special_chars(df)
#blob_sentiment=textblob_sentiment(df)
#df['blob_sentiment']=blob_sentiment
#df=removing_stop_words(df)
#df=spell_correction(df)


newDF = pd.DataFrame()
newDF['review']=df['review']
newDF['sentiment']=df['sentiment']

X,y,cv=bow_count_vector(newDF)
#X=np.concatenate([X1, X2], axis=1)

save_to_pikl(cv,"D:\Python\sentiment_analysis\pkl_files_1\cv_pikl.pkl")
X=np.c_[ X, df['upper_case_words'] ]

#X=np.c_[ X, df['stop_words'] ]
#X=np.c_[ X, df['word_count'] ]
#X=np.c_[ X, df['char_count'] ]
#X=np.c_[ X, df['avg_word_lengths'] ]
#X=np.c_[ X, df['blob_review'] ]

from sklearn.naive_bayes import MultinomialNB
mnb_clf = MultinomialNB(alpha=.6)
y_pred,mnb_clf=K_folds(X,y,mnb_clf,"mnb")

from sklearn.ensemble import AdaBoostClassifier
ada_bst_clf=AdaBoostClassifier(n_estimators =180,learning_rate=.6)
predY,ada_bst_clf=K_folds(X,y,ada_bst_clf,"adaboost")

import xgboost as xgb
clf_xgb = xgb.XGBClassifier(
 learning_rate = .6,
 n_estimators= 170)
y_pred,clf_xgb=K_folds(X,y,clf_xgb,"xgb")

from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(n_estimators =160, criterion = 'entropy', random_state = 10)
y_pred,et_clf=K_folds(X,y,et_clf,"et")

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
y_pred,et_clf=K_folds(X,y,et_clf,"et")

# =============================================================================
# import pickle
# # save the classifier
# with open('my_multinomialnb_classifier.pkl', 'wb') as fid:
#     cPickle.dump(clf, fid)
#     
# import pickle
# with open('my_multinomialnb_classifier.pkl', 'rb') as fid:
#     clf_pikl = pickle.load(fid)


# =============================================================================
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# 
# 
# from sklearn.ensemble import AdaBoostClassifier
# ada_bst_clf=AdaBoostClassifier(n_estimators =180,learning_rate=.6)
# ada_bst_clf.fit(X_train,y_train)
# y_pred=ada_bst_clf.predict(X_test)
# 
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test,y_pred)
# =============================================================================




# =============================================================================