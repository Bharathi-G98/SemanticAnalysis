import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from gensim.test.utils import get_tmpfile
from numba import jit, cuda
from timeit import default_timer as timer
# Import Data
df=pd.read_csv('D://SentAndSimilarity//questions.csv')
# Check for null values
df[df.isnull().any(axis=1)]
# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index,inplace=True)
question1=list(df['question1'].values)
question2=list(df['question2'].values)
qid1=list(df['qid1'].values)
qid2=list(df['qid2'].values)
is_dup=list(df['is_duplicate'].values)
# if remove_stopwords:
stops = set(stopwords.words("english"))

#@jit
#def all_func():
def clean(text):
    words = [w for w in text.lower().split() if not w in stops]
    final_text = " ".join(words)
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", final_text )
    review_text = re.sub(r"\'s", " 's ", final_text )
    review_text = re.sub(r"\'ve", " 've ", final_text )
    review_text = re.sub(r"n\'t", " 't ", final_text )
    review_text = re.sub(r"\'re", " 're ", final_text )
    review_text = re.sub(r"\'d", " 'd ", final_text )
    review_text = re.sub(r"\'ll", " 'll ", final_text )
    review_text = re.sub(r",", " ", final_text )
    review_text = re.sub(r"\.", " ", final_text )
    review_text = re.sub(r"!", " ", final_text )
    review_text = re.sub(r"\(", " ( ", final_text )
    review_text = re.sub(r"\)", " ) ", final_text )
    review_text = re.sub(r"\?", " ", final_text )
    review_text = re.sub(r"\s{2,}", " ", final_text )
    return review_text

# question1f=[]
# question2f=[]
# for i in question2:
#    question2f.append(clean(i))
# for i in question1:
#    question1f.append(clean(i))
labeled_questions=[]
for i in range(len(question1)):
    labeled_questions.append(TaggedDocument(question1[i].split(), [qid1[i]]))
    labeled_questions.append(TaggedDocument(question2[i].split(), [qid2[i]]))
model4 = Doc2Vec(dm = 1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
model4.build_vocab(labeled_questions)
# print('loading model')
# fname = get_tmpfile("my_doc2vec_model")
# #Run for first time)
# model2 = Doc2Vec.load(fname)
print('about to run epochs')
for epoch in range(15):
    model4.train(labeled_questions,epochs=model4.iter,total_examples=model4.corpus_count)
    print("Epoch #{} is complete.".format(epoch+1))
model4.save("d2vFinal4.model")

#run only this after training
model3= Doc2Vec.load("d2vFinal4.model")

from nltk.corpus import wordnet as wn


v1='That table is brown.'
v2='The table is brown.'

def check_vocab(word):
    syno=[]
    for syn in wn.synsets(word):
        for name in syn.lemma_names():
          if name not in syno and name!=word:
              syno.append(name)
              if name in model4.wv.vocab:
                return name
    return ''

def to_list(v1):
    v1=v1.split()
    for i in v1:
        if i not in model4.wv.vocab:
            rep=check_vocab(i)
            if (rep):
                v1[v1.index(i)]=rep
            else:
                del v1[v1.index(i)]
    return v1
v1="i like dogs"
v2="i adore dogs"
v1=to_list(v1)
v2=to_list(v2)
model4.n_similarity(v1,v2)




# v1 = model3.infer_vector(v1.split(), alpha=0.025, min_alpha=0.025, steps=50)
# v2 = model3.infer_vector(v2.split(), alpha=0.025, min_alpha=0.025, steps=50)
# v1=v1.reshape(1, -1)
# v2=v2.reshape(1, -1)
# spatial.distance.cosine(v1,v2)
# cosine_similarity(v1, v2)
