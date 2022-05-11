import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("fake_or_real_news.csv")
df.head()
x = df.loc[:,['text']]
y = df.label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
x_train_docs = [doc for doc in x_train.text]
pipeline = Pipeline([
    ('vect', TfidfVectorizer( ngram_range=(1,3), stop_words='english', max_features=1000)),
    ('svc', LinearSVC())
])
pipeline.fit(x_train_docs, y_train)
scores = cross_val_score(pipeline, x_train_docs, y_train, cv=5)
mean_cross_val_accuracy = np.mean(scores)
x_test_docs  =  [doc  for  doc  in  x_test.text]
y_test_pred = pipeline.predict(x_test_docs)
accuracy_score(y_test,y_test_pred)
pickle.dump(pipeline, open('pipeline.pkl', 'wb'))

x_docs = [doc for doc in x.text]
y_predict = pipeline.predict(x_docs)
df['Predict_label']=y_predict
df.to_csv('Predict_labeled.csv')
