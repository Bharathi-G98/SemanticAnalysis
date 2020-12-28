from flask import Flask, request
from gensim.models import Doc2Vec
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize
from statistics import mean
import string

app = Flask(__name__)

model3= Doc2Vec.load("d2vFinal4.model")
def check_vocab(word):
    syno=[]
    for syn in wn.synsets(word):
        for name in syn.lemma_names():
          if name not in syno and name!=word:
              syno.append(name)
              if name in model3.wv.vocab:
                return name
    return ''

def to_list(v1):
    v1=v1.split()
    for i in v1:
        if i not in model3.wv.vocab:
            rep=check_vocab(i)
            if (rep):
                v1[v1.index(i)]=rep
            else:
                del v1[v1.index(i)]
    return v1
def calc_sim(v1,v2):
    v1=to_list(v1)
    v2=to_list(v2)
    return str(model3.wv.n_similarity(v1,v2))




@app.route("/similarity",methods=['POST'])
def similarity():
    data=request.get_json()
    text1 = sent_tokenize(data['text1'])
    text2=sent_tokenize(data['text2'])
    maxSims=[]
    for s1 in text1:
        s1=s1.lower()
        s1=s1.translate(str.maketrans('', '', string.punctuation))
        sims=[]
        print(s1)
        for s2 in text2:
            s2=s2.lower()
            s2=s2.translate(str.maketrans('', '', string.punctuation))
            sims.append(float(calc_sim(s1,s2)))
        print(sims)
        maxSims.append(max(sims))
    # print(pairs)
    print(maxSims)
    return str(mean(maxSims))



    return calc_sim(sent1,sent2)

if __name__ == '__main__':
     app.run(port=5000)
