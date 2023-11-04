import sys


import pymongo

from pymongo import MongoClient

import numpy as np

import pandas as pd

import nltk


from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sentence_transformers import SentenceTransformer

# Load pre-trained BERT model

model = SentenceTransformer('bert-base-nli-mean-tokens')





client = MongoClient('mongodb://127.0.0.1:27017/')

db = client['organizationsDB']

collection=db['answers']

data = collection.find()

for document in data:

    #print(document['question']['maxMarks'])

    marks=0

    # Load model answer

    model_answer = document['question']['solution']
    model_answerForGPT=document['question']['solution']

    # Load student answer

    student_answer = document['answer']
    student_answerForGPT = document['answer']
    # Preprocess model answer and student answer

    stop_words = set(stopwords.words('english'))

    model_answer = ' '.join([word.lower() for word in model_answer.split() if word.lower() not in stop_words])

    student_answer = ' '.join([word.lower() for word in student_answer.split() if word.lower() not in stop_words])



    # Calculate LSA similarity score

    vectorizer = TfidfVectorizer()

    lsa = TruncatedSVD(n_components=1, n_iter=100)

    model_tfidf = vectorizer.fit_transform([model_answer, student_answer])

    model_lsa = lsa.fit_transform(model_tfidf)

    lsa_score = np.dot(model_lsa[0], model_lsa[1])



    # Calculate BERT similarity score

    model_embedding = model.encode([model_answer])

    #print(model_embedding)

    student_embedding = model.encode([student_answer])

    #print(student_embedding)

    bert_score = np.dot(model_embedding[0], student_embedding[0])

    outOf=np.dot(model_embedding[0],model_embedding[0])

    

    LsaPer=lsa_score*100

    BertPer=bert_score*100/outOf
    bert_scor = np.dot(student_embedding[0],model_embedding[0])

    outO=np.dot(student_embedding[0],student_embedding[0])

    BertPe=bert_scor*100/outO

    if ((LsaPer+BertPer)/2)<75:

        if((LsaPer+BertPer)/2)<60:

            marks=0

        else:

            marks=(((LsaPer+BertPer+BertPe)/3))*document['question']['maxMarks']/100

    elif BertPer<85:

        marks=(((BertPer-85)+25)/30)*document['question']['maxMarks']/100

    else:

        

        marks=((LsaPer+BertPer+BertPe)/3)*document['question']['maxMarks']/100

    print(round(marks,2))
    print("BertPercent= ",BertPer)
    print("LSAPer= ",LsaPer)
    collection.update_one({'_id': document['_id']}, {'$set': {'obtainMarks': marks}})

    
    

    

print("AutoEvaluated all answers Successfully")

#
#
#
#
#Integrating openAI

import openai

openai.api_key='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'
student_answerForGPT="Prime Minister of India is Narendra Modi"
model_answerForGPT="Narendra Modi is prime minister of India"
messages=[
        {"role": "system", "content": "You are a helpful assistant to evaluate the student answer from model answer and many krnowledge that you have ans then assign marks out of 100."}
    ]


message="Evaluate Student Answer with reference to model answer and your knowledge out of 100 Marks and return your reply in integer only after final evaluation. Student Answer: "+student_answerForGPT+". \n Model answer: "+model_answerForGPT

messages.append(
    {"role": "user", "content": message}
)
resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages
)
reply=resp.choices[0].message.content
print(f"Open AI evaluated as: {reply}")
messages.append({"role": "assistant", "content": reply})