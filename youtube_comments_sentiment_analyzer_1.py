import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Commented out IPython magic to ensure Python compatibility.
#Libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import nltk
# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation

import re

"""### **Read data**"""

data = pd.read_csv('comments.csv')
data.columns
data1=data.drop(['Unnamed: 0','Likes','Time','user','UserLink'],axis=1)
data1

"""### **Data labelling**"""

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data1["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data1["Comment"]]
data1["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data1["Comment"]]
data1["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data1["Comment"]]
data1['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data1["Comment"]]
score = data1["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data1["Sentiment"] = sentiment
data1.head()

"""### **Final data**"""

data2=data1.drop(['Positive','Negative','Neutral','Compound'],axis=1)
data2.head()

"""### **Data transformation**"""

stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer() 
snowball_stemer = SnowballStemmer(language="english")
lzr = WordNetLemmatizer()

"""## **Data Preprocessing**"""

def text_processing(text):   
    # convert text into lowercase
    text = text.lower()

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text

nltk.download('omw-1.4')
data_copy = data2.copy()
data_copy.Comment = data_copy.Comment.apply(lambda text: text_processing(text))

le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

processed_data = {
    'Sentence':data_copy.Comment,
    'Sentiment':data_copy['Sentiment']
}

processed_data = pd.DataFrame(processed_data)
processed_data.head()

processed_data['Sentiment'].value_counts()

"""### **Balancing data**"""

df_neutral = processed_data[(processed_data['Sentiment']==1)] 
df_negative = processed_data[(processed_data['Sentiment']==0)]
df_positive = processed_data[(processed_data['Sentiment']==2)]

# upsample minority classes
df_negative_upsampled = resample(df_negative, 
                                 replace=True,    
                                 n_samples= 205, 
                                 random_state=42)  

df_neutral_upsampled = resample(df_neutral, 
                                 replace=True,    
                                 n_samples= 205, 
                                 random_state=42)  


# Concatenate the upsampled dataframes with the neutral dataframe
final_data = pd.concat([df_negative_upsampled,df_neutral_upsampled,df_positive])

#final_data = processed_data

final_data['Sentiment'].value_counts()

corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

"""## **Machine learning model**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
nb_score = accuracy_score(y_test, y_pred)
print('nb accuracy',nb_score)

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

clf= classifier
vectorizer = CountVectorizer()

X_vect = vectorizer.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values
print(len(y))
# Train the classifier
clf = MultinomialNB()
clf.fit(X_vect, y)



# Test the classifier on a new input
new_X = ['This is a positive document']
new_X_vect = vectorizer.transform(new_X)

# Get the feature probabilities for each class
classes = clf.classes_
feature_prob = clf.feature_log_prob_
class_log_prior = clf.class_log_prior_

# Calculate the probability of the input document for each class
prob_total = np.zeros(len(classes))
for i in range(len(classes)):
    class_index = list(classes).index(classes[i])
    log_prob_total = np.sum(feature_prob[class_index] * new_X_vect.toarray()[0]) + class_log_prior[class_index]
    prob_total[i] = np.exp(log_prob_total)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()
feature_names_arr = np.array(feature_names)

# Print the results in the form of x*y/z for each feature and each class
for i, c in enumerate(classes):
    class_index = list(classes).index(c)
    prior_prob = np.exp(class_log_prior[class_index])
    prob_features_given_class = np.exp(feature_prob[class_index])
    print(f"P({c} | document) = P(document | {c}) * P({c}) / P(document)")
    for j, prob in enumerate(prob_features_given_class):
        feature = feature_names_arr[j]
        feature_prob_total = prob * prior_prob / prob_total[i]
        print(f"P({c} | {feature}) = {prob:.4f} * {prior_prob:.4f} / {prob_total[i]:.4f} = {feature_prob_total:.4f}")
    print()

"""### **Evaluation**"""

from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
svm_score = accuracy_score(y_test, y_pred)
print('svm accuracy',svm_score)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
knn_score = accuracy_score(y_test, y_pred)
print('knn accuracy',knn_score)

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

A = neigh.kneighbors_graph(X)
A.toarray()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=40)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(accuracy_score(y_train, y_pred_train))
print('random forest accuracy',score)

from sklearn.ensemble import BaggingClassifier
model2 = BaggingClassifier(base_estimator = classifier,n_estimators=300,random_state=2)
ans = model2.fit(X_train,y_train)
y_pred_train = ans.predict(X_train)
y_pred_test = ans.predict(X_test)
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))
bagging_score=accuracy_score(y_test,y_pred_test)

from sklearn.ensemble import VotingClassifier
estimator = []
estimator.append(('Naive Bayes', classifier))
estimator.append(('SVM', svm))
# estimator.append(('RandomForest',model))


model = VotingClassifier(estimators =estimator, voting='hard')
ans = model.fit(X_train,y_train)

y_pred_test = ans.predict(X_test)
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

