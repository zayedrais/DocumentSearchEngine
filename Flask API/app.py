import pandas as pd
import numpy as np
import os
import re
import operator
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import flask
import pickle
import json
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#

df_news = pd.read_csv('df_news_index.csv')
with open("vocabulary_news20group.txt", "r") as file:
    vocabulary = eval(file.readline())

Tfidmodel =pickle.load(
    open('/home/zettadevs/Deploy_TFID_model/tfid.pkl', 'rb'))
traineddata = Tfidmodel.A #np.float16(Tfidmodel.A)



def wordLemmatizer(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k = pd.DataFrame()
    for index, entry in enumerate(data):

        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                
    return file_clean_k

## Create vector for Query/search keywords


def gen_vector_T(tokens, tfidf):

    Q = np.zeros((len(vocabulary))) 
    x = tfidf.transform(tokens)
    for token in tokens[0].split(','):
        try:
            ind = vocabulary.index(token)
            Q[ind] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#-------- ROUTES GO HERE -----------#
@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/Search', methods=["GET"])
def DrugFind():
    # embed the query for calcluating the similarity
    query = flask.request.args.get('query')
    #print(query)
    preprocessed_query = preprocessed_query = re.sub(
        "\W+", " ", query.lower()).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    q_df = q_df.replace(to_replace="'", value='', regex=True)
    q_df = q_df.replace(to_replace="\[", value='', regex=True)
    q_df = q_df.replace(to_replace=" ", value='', regex=True)
    q_df = q_df.replace(to_replace='\]', value='', regex=True)

    d_cosines = []
    tfidf = TfidfVectorizer(vocabulary=vocabulary , dtype=np.float32)
    tfidf.fit(q_df['q_clean'])
    query_vector = gen_vector_T(q_df['q_clean'], tfidf)
    #query_vector = np.float16(query_vector)
    for d in traineddata:
        d_cosines.append(cosine_sim(query_vector, d))
    out = np.array(d_cosines).argsort()[-10:][::-1]
   
    d_cosines.sort()

    a = pd.DataFrame()
    for i, index in enumerate(out):
        a.loc[i, 'Subject'] = df_news['Subject'][index]
        #a.loc[i, 'rating'] = df_webmd['rating'][index]
    for j, simScore in enumerate(d_cosines[-10:][::-1]):
        a.loc[j, 'Score'] = simScore
    a = a.sort_values(by='Score', ascending=False)
    js = a.to_json(orient='index')
    js =js.replace('[', '').replace(']', '')
    ls = js.split('},')

    l = [re.sub(r'\"[0-9]\":', '', l) for l in ls]
    l[0] = re.sub(r'^{{1}', '', l[0])      
    l = [re.sub(r'^,{1}', '', l) for l in l]
    l = [ls+'}' for ls in l]
    l[9] = l[9].replace('}}', '')
    lsDrug =[]
    for txt in l:
        tx =json.loads(txt)
        lsDrug.append(tx)
    # response = app.response_class(
    #     response=json.dumps(lsDrug),
    #     status=200,
    #     mimetype='application/json'
    # )
    return flask.jsonify(lsDrug) 


#if __name__ == '__main__':
    #'Connects to the server'
    #HOST = '127.0.0.1'
    #PORT = 5000      #make sure this is an integer

    #export FLASK_ENV=development
#    app.run()
