from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField
from flask import render_template,Flask, request
from wtforms.validators import DataRequired
# Misc
import json
import re
import numpy as np
import editdistance
from collections import Counter
import pickle

# Scikit
from sklearn.preprocessing import MultiLabelBinarizer
# Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# load LSTM model
model = load_model("./data/lstm_model_100.h5")
model._make_predict_function()
# tokenizer load
with open('./data/prothom_alo_100_tk.pickle', 'rb') as handle:
    tk = pickle.load(handle)
# multilabelbinarizer load
with open('./data/prothom_alo_100_mlb.pickle', 'rb') as handle:
    mlb = pickle.load(handle)
    labels = mlb.classes_
with open('./data/stopwords-bn.txt','r') as txtfile:
    lines = txtfile.readlines()
with open('./data/all_tags.data', 'rb') as filehandle:
    # read the data as binary data stream
    all_tags = pickle.load(filehandle)

stopword_bn = [i.strip() for i in lines]
pattern = re.compile(u'[(.$)):]|[,।‘’-]|\s\s+')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thesis2020'
run = 0
def rmv_punc(text):
    if isinstance(text,str):
        return pattern.sub(' ',text).strip()
def rmv_stopword(text):
    if isinstance(text,str):
        return ' '.join([w for w in text.split() if w not in stopword_bn])
def stem(text):
    global run
    if isinstance(text,str):
        if (run%10000)==0:
            print("Running %d th article" %(run))
        run+=1
        return ' '.join([stemmer.stem_word(w) for w in text.split()])
def same_word(w1,w2):
    """
    Measures word similarity based on EditDistance Algorithm on the last indices of words.
    """
    dist = editdistance.eval(w1,w2)
    if len(w1)>2 and len(w2)>2 and dist<=6: # 6 is the length of গুলোতে, longest bibhokti
        
        t1 = w1[0:int(len(w1)/2)+1] # cutting in half
        t2 = w2[0:int(len(w1)/2)+1]
        dist2 = editdistance.eval(t1,t2)
        if dist2==0: # matching if first half of the words are same
            return True
    return False

def match_w_tag_bank(temp_tags):
    new_list = []
    for w1 in all_tags:
        for w2 in range(len(temp_tags)):
            if same_word(w1,temp_tags[w2]):
                temp_tags[w2] = w1
    return temp_tags

def suggest(*article):
    title = rmv_punc(rmv_stopword(article[0]))
    text = rmv_punc(rmv_stopword(article[1]))

    # extracting tags from title
    if title.strip() != "":
        title_words = [w for w in title.split()]
        content_words = [w for w in text.split()]
        x = list()
        for t in range(len(title_words)):
            for c in range(len(content_words)):
                if title_words[t] == content_words[c]:
                    x.append(title_words[t])
                if same_word(title_words[t],content_words[c]):
                    if(len(title_words[t])<=len(content_words[c])):
                        x.append(title_words[t])
                    else:
                        x.append(title_words[t])

        counter = Counter(x)
        tag_candidates = [w[0] for w in counter.most_common(3) if w[1]>=3]
        temp = match_w_tag_bank(tag_candidates)


    # predict tags from trained model
    maxlen = 200
    text = rmv_punc(text)
    text = rmv_stopword(text)
    text = tk.texts_to_sequences([text])
    text = pad_sequences(text,maxlen=maxlen)
    text = np.array(text)
   
    pred_labels = model.predict([text])[0]
    tag_prob = dict([(labels[i], prob) for i, prob in enumerate(pred_labels.tolist())])
    pred = sorted(tag_prob.items(), key=lambda kv: kv[1],reverse=True)[:3]
    tag_proper = [w[0] for w in pred]
    return temp+tag_proper

class MyForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    text = StringField('Text', validators=[DataRequired()])

@app.route("/")
def index():
    form = MyForm()
    return render_template("reviewform.html", form=form)


@app.route("/results", methods=["POST"])
def results():
    form = MyForm(request.form)
    if request.method == "POST":
        title = request.form["title"]
        text = request.form["text"]
    else:
        title = "[None Submitted]"
        text = "[None submitted]"
    tags = suggest(title, text)
    return render_template("results.html",title=title,text=text,tags=tags)
    return render_template("reviewform.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)
