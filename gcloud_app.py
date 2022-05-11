import streamlit as st
import pandas as pd
from streamlit_player import st_player
from datetime import datetime
import numpy as np
import seaborn as sns
from google.cloud import firestore
import joblib
import base64
import matplotlib.pyplot as plt
import pickle
import scipy
import tempfile

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "aml-final-team-project-c8fc9fde12f6.json"
db = firestore.Client(project='aml-final-team-project')

st.set_page_config(layout="wide", page_title="Fake And Real News Predictor", page_icon="🤗")

st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

#################################################################
##################### App Header ################################
#################################################################
st.title('Fake And Real News Prediction')
st.markdown(
    """
Predict the truth of thousands of news.
Using our predictor, it's easy to tell the real news from the fake.  
"""
)
#################################################################
###################### home page ################################
#################################################################

select_bar=st.sidebar.radio('Menu',['Home','Model','About'])


if select_bar=='Home':
    st.header('Home Page')
    st.subheader('Find In Real World')
    st.write('In the era of information explosion, people receive all kinds of news from various social media every day, and it is difficult for us to distinguish the truth from the fake news.')
    st.markdown('---')
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader('Fake News vs Real News')
        st.write('When people use social networks, it is difficult for them to subjectively judge the truth or falsity of the news they receive. On the other hand, people rarely question the truthfulness of the news, and they share the fake news with their family and friends, thus causing the spread of false news.')
        st.write('From past data, 23% of social media users admit to sharing false political news either intentionally or accidentally. ')
        st.write('Secondly, fake news is not only spread by humans, but also by computer-programmed social media accounts that automatically post fake news on the web, allowing fake news to spread among followers.')
        st.write('Related Paper: https://rdcu.be/cMvdh')


    with col2:
        st.subheader('How fake news affect our life.')
        st_player("https://youtu.be/irHP3znqwq8")
        with st.expander('Video Credit'):
            st.write("""
            Creator: CBS News 

            URL: https://www.youtube.com/watch?v=irHP3znqwq8
        """)
    st.markdown('---')

    st.subheader('What\'s our app doing?')
    
    st.write('In real life, people can receive different kinds of news from various sources every second. Due to the increasing number and decreasing quality of news, it is difficult for the public to get useful information from the large amount of news - that is, to distinguish between real and fake news. Our application aims to help the public distinguish between real and fake news and get useful information from the mass of information.')
    st.write('By importing news data, including news headlines and news sources, our application will determine whether the news is true or false.')
    st.markdown('---')



    st.subheader('Sample data from reddit')
    st.write('In this section we will show our model performance by predicting posts from reddit. We chose to extract text from following two subreddit - r/news and r/fakenews and use our model to predict the result')
    option = st.selectbox(
     'Check the number of Real or Fake News in the following subreddit: Please select the news type you want to check',
     ('real', 'fake'))
    docs = db.collection(option+'_news').stream()
    items = []
    for doc in docs:
        items.append(doc.to_dict())
    df = pd.DataFrame.from_records(items)
    df['mmmdd'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%b %d'))
    df['color'] = df['sentiment'].apply(lambda x: 'orange' if x == 'FAKE' else 'skyblue')
    ############# Filters ######################################
    
    ######### Date range slider ################################
    start, end = st.sidebar.select_slider(
                    "Select Date Range", 
                    df.mmmdd.drop_duplicates().sort_values(), 
                    value=(df.mmmdd.min(), df.mmmdd.max()))
    df_filter = df.loc[(df.mmmdd >= start) & (df.mmmdd <= end), :]
    if df_filter.shape[0] > 0:
        ######### Main Story Plot ###################################
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(df_filter.mmmdd, df_filter.sentiment).plot(
                    kind="bar", 
                    figsize=(6,2), 
                    xlabel = "Date",
                    color={'REAL':'skyblue', 'FAKE': 'orange'})
            st.pyplot(ax.figure)
        with col2:
            st.write('This plot shows the daily count of real and fake in the '+option+' news subreddit.')
        st.markdown('---')
        ######## Sample Reviews and Sentiment Predictions ###############
        st.subheader("Sample posts and Predictions")
        df_sample = df_filter.sample(5)
        for index, row in df_sample.iterrows():
            col1, col2 = st.columns((1,5))
            with col1:
                if row['sentiment'] == "REAL":
                    st.success("REAL") 
                else: 
                    st.error("FAKE")    
            with col2:
                with st.expander(row['title']):
                    st.write(row['comment'])
    else:
        st.warning("Your selection returned no data. Change your selecton.")

#################################################################
##################### Model Page ################################
#################################################################
elif select_bar=='Model':
    st.header('Model Page')
    st.markdown('---')
    st.subheader('Model Selection')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Tokenization is the most important step in building a model to identify fake or real news. We’ve tried various tokenization method including countvectorizer, tdidf, word_tokenizer by nltk. Customized tokenizer took us the most time when implementing it in the model, but the model built by customized tokenizer didn’t improve the test accuracy. ')
        st.write('From all the combinations of the tokenizer and the model, tdidf was our best choice. Three classifiers were discussed when building the best model. Multinomial NB is the classifier that was used most frequently in other people’s work. However, the training and testing accuracy has a large gap which seemed concerning to us, thus it was abandoned at first. Passiveaggressive classifier was the one gave us the best training accuracy while its testing accuracy wasn’t as ideal as linearsvc’s testing accuracy. We chose linearsvc as our classifier for the model.')
    with col2:
        st.image('Model Performance.png')
    st.markdown('---')
    st.subheader('Wordcloud of training dataset')
    col1, col2 = st.columns(2)
    with col1:
        st.image('wordcloud_text.png')
        st.caption('Title wordcloud')
    with col2:
        st.image('wordcloud_title.png')
        st.caption('Text wordcloud')
    st.subheader('How to build the model and pipeline.')

    st.write("Fake news is false or misleading information presented as news. And fake news damages the reputation of an individual or entity that then it makes money through advertising revenue. So we want people can to distinguish what is real news and fake news.Let people not be fooled by fake news. ")
    df = pd.read_csv('fake_or_real_news.csv',index_col=[0]).reset_index(drop=True)
    st.write('This dataset is not missing value.')
    x =df.isna().sum()
    x
    st.subheader("Is our dataset imbalance or balance?")
    col1, col2= st.columns(2)
    with col1:
       colors=['blue','orange']
       fig_sb, ax_sb = plt.subplots()
       plt.pie(df['label'].value_counts(),labels=['REAL','FAKE'],autopct='%.2f%%',explode=[0.01,0.01],colors=colors);
       plt.ylabel('Fake News vs Real News')
       plt.figure(figsize=(4,2))
       st.pyplot(fig_sb)
    with col2:
       st.write("")
       st.write("")
       st.write("") 
       st.subheader("We can see our dataset is balance.")

    with st.expander("model"):
        code ='''
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
#data
df = pd.read_csv("fake_or_real_news.csv")
df.head()
x = df.loc[:,['text']]
y = df.label
#X,y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
x_train_docs = [doc for doc in x_train.text]
#final model:
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
'''  
        st.code(code, language='python')
    st.write("You can get use your dataset to predict the news is fake or real.")
    #Load pipeline
    
    pipe = pickle.load(open('pipeline.pkl', 'rb'))
    title1 = st.text_input("You text your news' title:", )
    text1 = st.text_area('You text your news:', )
    df1 = pd.DataFrame([{'title1':title1,'text1':text1}])
    b=len(title1)
    c=len(text1)
    if st.button('Predict') and (b+c>0):
        predictions = pipe.predict(df1)
        a= predictions
        if (a[0]=='FAKE')& (a[1]=='FAKE'):
            st.info('This news is FAKE NEWS')
            st.snow() 
        elif (a[0]=='REAL') &(a[1]=='REAL'):
            st.info('This news is FAKE NEWS')
            st.balloons()
    else:
        st.write("You need type something.")
            
#################################################################
##################### About Page ################################
#################################################################
elif select_bar=='About':
    st.subheader('This is our team.')
    st.write('team member: Haoxuan Wei, Jiawei Chen, Jingjing Huang, Yuxin Tan, Ziyan Wang')
    col1,col2,col3=st.columns(3)
    with col1:     
        st.image('member1.jpg')
        st.caption('Team member: Haoxuan Wei')
    with col2:
        st.image('member2.jpg')
        st.caption('Team member: Jiawei Chen')
    with col3:
        st.image('member3.jpg')
        st.caption('Team member: Jingjing Huang')
    col1,col2=st.columns(2)   
    with col1:
        st.image('member4.jpg')
        st.caption('Team member: Yuxin Tan')
    with col2:
        st.image('member5.jpg')
        st.caption('Team member: Ziyan Wang')
