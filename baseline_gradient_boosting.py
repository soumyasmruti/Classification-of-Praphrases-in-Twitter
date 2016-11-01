
# coding: utf-8

# In[1]:

import csv
import pandas as pa
import numpy as np 
import scipy
pa.options.mode.chained_assignment = None
import sys  
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import distance
import random

import nltk

from collections import *
from collections import Counter
import short_sentence_similarity as ss
from itertools import izip, islice, tee

reload(sys)  
sys.setdefaultencoding('utf8')

from nltk.tokenize import WordPunctTokenizer
w = WordPunctTokenizer()

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

from nltk import stem
english_stemmer = stem.SnowballStemmer('english')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[2]:

train = pa.read_csv("./data/train.data", sep='\t',  header= None, names=['trendid', 'trendname', 'origsent', 'candsent', 'judge', 'origsenttag', 'candsenttag'])
test = pa.read_csv("./data/test.data", sep='\t', header=None, names=['trendid', 'trendname', 'origsent', 'candsent', 'label', 'origsenttag', 'candsenttag'])
test_labels =  pa.read_csv("./data/test.label", sep='\t', header=None, names = ['ans', "prob"])

# In[3]:

def functrain(row):
    if eval(row['judge'])[0] >= 3:
        return 1
    elif eval(row['judge'])[0] == 2:
        return 2
    else:
        return 0

    
def functest(row):
    if int(row['label']) in (4,5):
        return 1
    elif int(row['label']) == 3:
        return 2
    else:
        return 0
train["is_paraphrase"] = train.apply(functrain, axis=1)
test["is_paraphrase"] = test.apply(functest, axis=1)
train = train[train["is_paraphrase"] != 2]
test = test[test["is_paraphrase"] != 2]
#train.append(ppdb)
ytrain = train.pop("is_paraphrase")
ytest = test.pop("is_paraphrase")
train.drop('judge', axis=1, inplace=True)
test.drop('label', axis=1, inplace=True)


# In[4]:

#######################
## N-Gram Extraction ##
#######################
def getUnigram(words):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of unigram
    """
    assert type(words) == list
    return words
    
def getBigram(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ['I', 'am', 'Denny']
       Output: a list of bigram, e.g., ['I_am', 'am_Denny']
       I use _ as join_string for this example.
    """
    if type(words) == tuple:
        words = words[1]
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = getUnigram(words)
    return lst
    
def getTrigram(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of trigram, e.g., ['I_am_Denny']
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = getBigram(words, join_string, skip)
    return lst
    

#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def preprocess_data(line):
    tokens = [word.lower() for word in w.tokenize(line) if word.isalpha()]
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    #tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

def preprocess(line):
    return " ".join(preprocess_data(line))

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


# In[5]:


def extract_count_feat(df):
    
    ################################
    ## word count and digit count ##
    ################################
    print "generate word counting features"
    feat_names = ["origsent", "candsent"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    
    for stem in ["", "_stem"]:
        for feat_name in feat_names:
            for gram in grams:
                ## word count
                df["count_of_%s_%s%s"%(feat_name,gram, stem)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram+stem]), axis=1))
                df["count_of_unique_%s_%s%s"%(feat_name,gram, stem)] = list(df.apply(lambda x: len(set(x[str(feat_name+"_"+gram+stem)])), axis=1))
                df["ratio_of_unique_%s_%s%s"%(feat_name,gram, stem)] = map(try_divide, df["count_of_unique_%s_%s%s"%(feat_name,gram, stem)], 
                                                                           df["count_of_%s_%s%s"%(feat_name,gram,stem)])

            ## digit count
            df["count_of_digit_in_%s%s"%(feat_name, stem)] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"+stem]), axis=1))
            df["ratio_of_digit_in_%s%s"%(feat_name, stem)] = map(try_divide, df["count_of_digit_in_%s%s"%(feat_name, stem)], df["count_of_%s_unigram%s"%(feat_name, stem)])

    ## origsent missing indicator
    df["origsent_missing"] = list(df.apply(lambda x: int(x["origsent_unigram"] == ""), axis=1))
    df["candsent_missing"] = list(df.apply(lambda x: int(x["candsent_unigram"] == ""), axis=1))


    ##############################
    ## intersect word count ##
    ##############################
    
    #### Count & Ratio of a’s n-gram in b’s n-gram 
    print "generate intersect word counting features"
    #### unigram
    for stem in ["", "_stem"]:
        for gram in grams:
            for obs_name in feat_names:
                for target_name in feat_names:
                    if target_name != obs_name:
                        ## query
                        df["count_of_%s_%s_in_%s%s"%(obs_name,gram,target_name,stem)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram+stem] if w in set(x[target_name+"_"+gram+stem])]), axis=1))
                        df["ratio_of_%s_%s_in_%s%s"%(obs_name,gram,target_name,stem)] = map(try_divide, df["count_of_%s_%s_in_%s%s"%(obs_name,gram,target_name,stem)], df["count_of_%s_%s%s"%(obs_name,gram,stem)])



    ######################################
    ## intersect word position feat ##
    ######################################
    #### Statistics of Positions of a’s n-gram in b’s n-gram 
    #### For those intersect n-gram, I recorded their positions, and computed the following statistics as features. 
    #     – minimum value (0% quantile) 
    #     – median value (50% quantile) 
    #     – maximum value (100% quantile) 
    #     – mean value 
    #     – standard deviation (std) 
    #### Statistics of Normalized Positions of a’s n-gram in b’s n-gram 
    #### These features are similar with above features, but computed using positions normalized by the length of a.

    print "generate intersect word position features"
    for stem in ["", "_stem"]:
        for gram in grams:
            for target_name in feat_names:
                for obs_name in feat_names:
                    if target_name != obs_name:
                        pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                        ## stats feat on pos
                        df["pos_of_%s_%s_in_%s_min%s" % (obs_name, gram, target_name, stem)] = map(np.min, pos)
                        df["pos_of_%s_%s_in_%s_mean%s" % (obs_name, gram, target_name, stem)] = map(np.mean, pos)
                        df["pos_of_%s_%s_in_%s_median%s" % (obs_name, gram, target_name, stem)] = map(np.median, pos)
                        df["pos_of_%s_%s_in_%s_max%s" % (obs_name, gram, target_name, stem)] = map(np.max, pos)
                        df["pos_of_%s_%s_in_%s_std%s" % (obs_name, gram, target_name, stem)] = map(np.std, pos)
                        ## stats feat on normalized_pos
                        df["normalized_pos_of_%s_%s_in_%s_min%s" % (obs_name, gram, target_name, stem)] = map(try_divide, df["pos_of_%s_%s_in_%s_min%s" % (obs_name, gram, target_name, stem)], df["count_of_%s_%s%s" % (obs_name, gram, stem)])
                        df["normalized_pos_of_%s_%s_in_%s_mean%s" % (obs_name, gram, target_name, stem)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean%s" % (obs_name, gram, target_name, stem)], df["count_of_%s_%s%s" % (obs_name, gram, stem)])
                        df["normalized_pos_of_%s_%s_in_%s_median%s" % (obs_name, gram, target_name, stem)] = map(try_divide, df["pos_of_%s_%s_in_%s_median%s" % (obs_name, gram, target_name, stem)], df["count_of_%s_%s%s" % (obs_name, gram, stem)])
                        df["normalized_pos_of_%s_%s_in_%s_max%s" % (obs_name, gram, target_name, stem)] = map(try_divide, df["pos_of_%s_%s_in_%s_max%s" % (obs_name, gram, target_name, stem)], df["count_of_%s_%s%s" % (obs_name, gram, stem)])
                        df["normalized_pos_of_%s_%s_in_%s_std%s" % (obs_name, gram, target_name, stem)] = map(try_divide, df["pos_of_%s_%s_in_%s_std%s" % (obs_name, gram, target_name, stem)] , df["count_of_%s_%s%s" % (obs_name, gram, stem)])

# Jaccard coefficient
# JaccardCoef(A,B) = |A∩B|/|A∪B|

# and Dice distance
# DiceDist(A,B) = 2|A∩B|/|A|+|B|

# are used as distance metrics, where A and B denote two sets respectively. 
# For each distance metric, two types of features are computed. 
def extract_basic_distance_feat(df):
    ## jaccard coef/dice dist of n-gram
    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["origsent", "candsent"]
    for stem in ["", "_stem"]:
        for dist in dists:
            for gram in grams:
                for i in range(len(feat_names)-1):
                    for j in range(i+1,len(feat_names)):
                        target_name = feat_names[i]
                        obs_name = feat_names[j]
                        df["%s_of_%s_between_%s_%s%s"%(dist,gram,target_name,obs_name, stem)] = list(df.apply(lambda x: compute_dist(x[target_name+"_"+gram+stem], x[obs_name+"_"+gram+stem], dist), axis=1))
   
    print "generate rest all features"
    gram_ext = ["_unigram", "_bigram", "_trigram", "_char_unigram", "_char_bigram", "_char_trigram"]
    for stem in ["", "_stem"]:
        for gram in gram_ext:
            df["levenshtein_%s%s"%(gram,stem)] = list(df.apply(lambda x: distance.nlevenshtein(x["origsent"+gram+stem], x["candsent"+gram+stem], method=2) , axis=1))
            df["sorensen_%s%s"%(gram,stem)] = list(df.apply(lambda x: distance.sorensen(x["origsent"+gram+stem], x["candsent"+gram+stem]), axis=1))
            df["cosine_%s%s"%(gram,stem)] = list(df.apply(lambda x: cosine(x["origsent"+gram+stem], x["candsent"+gram+stem]), axis=1))
            df["precision_%s%s"%(gram,stem)] = list(df.apply(lambda x: precision_recall(x["origsent"+gram+stem], x["candsent"+gram+stem], x["origsent"+gram+stem]), axis=1))
            df["recall1gram_%s%s"%(gram,stem)] = list(df.apply(lambda x: precision_recall(x["origsent"+gram+stem], x["candsent"+gram+stem], x["candsent"+gram+stem]), axis=1))
            df["f1gram_%s%s"%(gram,stem)] = list(df.apply(lambda x: fmeasure(x["precision_%s%s"%(gram,stem)], x["recall1gram_%s%s"%(gram, stem)]), axis=1))
    
# The below function created TF-IDF matrix for the columns origsent and candsent. This may also be called 
# Bag of Words feature extraction.
nonnumeric_columns = set(['trendname', 'origsenttag', 'candsenttag', 'origsent_unigram', 'candsent_unigram',
                      'candsent', 'origsent', "origsent_tag", "candsent_tag", 'candsent_stem', 'origsent_stem',
                      'origsent_bigram', 'candsent_bigram', 'origsent_trigram', 
                      'candsent_trigram', 'origsent_tag_bigram', 'candsent_tag_bigram', 'origsent_tag_trigram', 
                      'candsent_tag_trigram', "origsent_tag_unigram", "candsent_tag_unigram",
                      'origsent_unigram_stem',
                      'candsent_unigram_stem', 'origsent_bigram_stem', 'candsent_bigram_stem', 
                      'origsent_trigram_stem', 'candsent_trigram_stem', 'origsent_char_unigram', 
                      'origsent_char_bigram', 'origsent_char_trigram', 'candsent_char_unigram',
                      'candsent_char_bigram', 'candsent_char_trigram', 'origsent_char_unigram_stem', 
                      'origsent_char_bigram_stem', 'origsent_char_trigram_stem', 'candsent_char_unigram_stem',
                      'candsent_char_bigram_stem', 'candsent_char_trigram_stem'])


def vectorize(train, tfv_query=None):
    #TF-IDF Calculation 
    desc_data = list(train['origsent'].apply(preprocess))
    legdesc_data = list(train['candsent'].apply(preprocess))
    if tfv_query is None:
        tfv_query = TfidfVectorizer(min_df=3,  max_features=None,   
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = stopwords)

        full_data = desc_data + legdesc_data
        tfv_query.fit(full_data)   
    
    
    
    # XGBoost(discussed below) doesn't (yet) handle categorical features automatically, so we need to change
    # them to columns of integer values.
    # See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
    # details and options
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        train[feature] = le.fit_transform(train[feature])
    
    train.drop(train.select_dtypes(include=['object']).columns, axis=1, inplace=True)
    train = train._get_numeric_data()
    features = list(train.columns[1:])
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    csr_train = sparse.csr_matrix(train.values)

#     Hstack all features along with tfidf features.
    return sparse.hstack([tfv_query.transform(desc_data), tfv_query.transform(legdesc_data), csr_train]), tfv_query


# In[6]:

def preprocess_token(line):
    tokens = [word.lower() for word in w.tokenize(line)]
    return tokens

def preprocess_data2(line):
    tokens = [word for word in w.tokenize(line.lower()) if word.isalpha()]
    return tokens

def cosine(a,b):
    # count word occurrences
    a_vals = Counter(a)
    b_vals = Counter(b)
    
    # convert to word-vectors
    words  = list(set(a_vals) | set(b_vals))
    a_vect = [a_vals.get(word, 0) for word in words]        
    b_vect = [b_vals.get(word, 0) for word in words]        
    
    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5             
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5             
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    
    cosine = dot / (len_a * len_b) 
    return cosine

def intersect (list1, list2) :
    cnt1 = Counter(list1)
    cnt2 = Counter(list2)   
    inter = cnt1 & cnt2
    return list(inter.elements())

def precision_recall(a, b, c):
    return len(set(intersect(a, b))) / len(set(c))

def fmeasure(a, b):
    if (a + b) > 0:
        return 2 * a * b / (a + b)
    return 0

def char_grams(b, gram):
    gramsD = {"unigram":1, "bigram":3, "trigram":3}
    n = gramsD[gram]
    return [b[i:i+n] for i in range(len(b)-n+1)]
    
def word2ngrams(text, gram, exact=True):
    """ Convert text into character ngrams. """
    gramsD = {"unigram":1, "bigram":3, "trigram":3}
    n = gramsD[gram]
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]
    
def preprocess_Event(line):
    tags = [word.lower().split('/',)[4] for word in line.split(' ',) if word.lower().split('/',)[4] != "o"]
    return tags
    
def preprocess_tag(line):
    tags = [word.lower().split('/',)[2] for word in line.split(' ',)]
    return tags

def word_left(word_list, tag_list):
    left_list = []
    if(len(word_list) == len(tag_list)):
        for word in word_list:
            idx = word_list.index(word)
            if idx > 0:
                left_list.append((word, tag_list[idx-1]))
            else:
                left_list.append((word, "phi"))
    return left_list
                
def word_right(word_list, tag_list):
    right_list = []
    if(len(word_list) == len(tag_list)):
        for word in word_list:
            idx = word_list.index(word)
            if idx < len(tag_list) -1:
                right_list.append((word, tag_list[idx-1]))
            else:
                right_list.append((word, "phi"))
    return right_list

def preprocess_NER(line):
    tags = [word.lower().split('/',)[1] for word in line.split(' ',) if word.lower().split('/',)[1] != "o" ]
    return tags

def feature_extraction(features):
    
    try:
        features["origsent_unigram"] = list(features.apply(lambda x: preprocess_token(x["origsent"]), axis=1))
    except:
        features["origsent_unigram"] = list(features.apply(lambda x: preprocess_data2(x["origsent"]), axis=1))
    try:
        features["candsent_unigram"] = list(features.apply(lambda x: preprocess_token(x["candsent"]), axis=1))
    except:
        features["candsent_unigram"] = list(features.apply(lambda x: preprocess_data2(x["candsent"]), axis=1))
        
    features["origsent_unigram_stem"] = list(features.apply(lambda x: preprocess_data(x["origsent"]), axis=1))
    features["candsent_unigram_stem"] = list(features.apply(lambda x: preprocess_data(x["candsent"]), axis=1))
    features["origsent_stem"] = list(features["origsent"].apply(preprocess))
    features["candsent_stem"] = list(features["candsent"].apply(preprocess))


    print "generate bigram"
    join_str = "_"
    try:
        features["origsent_bigram"] = list(features.apply(lambda x: getBigram(x["origsent_unigram"], join_str), axis=1))
    except:
        templist = []
        for x in features["origsent_unigram"].iteritems():
            templist.append(getBigram(x, join_str))
        features["origsent_unigram"] = templist
        
    try:
        features["origsent_bigram_stem"] = list(features.apply(lambda x: getBigram(x["origsent_unigram_stem"], join_str), axis=1))
    except:
        templist = []
        for x in features["origsent_unigram_stem"].iteritems():
            templist.append(getBigram(x, join_str))
        features["origsent_unigram_stem"] = templist
    
    features["candsent_bigram"] = list(features.apply(lambda x: getBigram(x["candsent_unigram"], join_str), axis=1))
    features["candsent_bigram_stem"] = list(features.apply(lambda x: getBigram(x["candsent_unigram_stem"], join_str), axis=1))
    ## trigram
    print "generate trigram"
    join_str = "_"
    features["origsent_trigram"] = list(features.apply(lambda x: getTrigram(x["origsent_unigram"], join_str), axis=1))
    features["candsent_trigram"] = list(features.apply(lambda x: getTrigram(x["candsent_unigram"], join_str), axis=1))
    features["origsent_trigram_stem"] = list(features.apply(lambda x: getTrigram(x["origsent_unigram_stem"], join_str), axis=1))
    features["candsent_trigram_stem"] = list(features.apply(lambda x: getTrigram(x["candsent_unigram_stem"], join_str), axis=1))
    
    #print "Generate Wordnet Features"
    #features["wordnet-similarity"] = list(features.apply(lambda x: ss.similarity(x["origsent"], x["candsent"], False), axis=1))
    #features["wordnet-similarity-norm"] = list(features.apply(lambda x: ss.similarity(x["origsent"], x["candsent"], True), axis=1))
    
    
    print "generate char gram"
    feat_names = ["origsent", "candsent"]
    grams = ["unigram", "bigram", "trigram"]
    for stem in ["", "_stem"]:
        for feat in feat_names:
            for gram in grams:
                try:
                    features["%s_char_%s%s"%(feat,gram,stem)] = list(features.apply(lambda x: word2ngrams(x[feat+stem], gram), axis=1))
                except:
                    continue
                nonnumeric_columns.add("%s_char_%s%s"%(feat,gram,stem))
    
    features["candsent_char_trigram"] = list(features.apply(lambda x: word2ngrams(x["candsent"], "trigram"), axis=1))
    features["origsent_char_bigram_stem"] = list(features.apply(lambda x: word2ngrams(x["candsent_stem"], "bigram"), axis=1))
    features["origsent_char_trigram_stem"] = list(features.apply(lambda x: word2ngrams(x["candsent_stem"], "trigram"), axis=1))
    
    print "generate common word features"
    gram_ext = ["_unigram", "_bigram", "_trigram", "_char_unigram", "_char_bigram", "_char_trigram"]
    for stem in ["", "_stem"]:
        for gram in gram_ext:
            features["common-words_%s%s"%(gram,stem)] = list(features.apply(lambda x: len(intersect(x["origsent"+gram+stem], x["candsent"+gram+stem])) , axis=1))
            
    features["origsent_tag"] =  list(features.apply(lambda x: preprocess_tag(x["origsenttag"]), axis=1))
    features["candsent_tag"] =  list(features.apply(lambda x: preprocess_tag(x["candsenttag"]), axis=1))
    
    features["origsent_tag_unigram"] =  features["origsent_tag"]
    features["candsent_tag_unigram"] =  features["candsent_tag"]
    
    features["origsent_tag_left"] = list(features.apply(lambda x: word_left(x["origsent_unigram"], x["origsent_tag"]), axis=1))
    features["candsent_tag_left"] = list(features.apply(lambda x: word_left(x["candsent_unigram"], x["candsent_tag"]), axis=1))

    features["origsent_tag_right"] = list(features.apply(lambda x: word_right(x["origsent_unigram"], x["origsent_tag"]), axis=1))
    features["candsent_tag_right"] = list(features.apply(lambda x: word_right(x["candsent_unigram"], x["candsent_tag"]), axis=1))
    
    features["origsent_NER"] =  list(features.apply(lambda x: preprocess_NER(x["origsenttag"]), axis=1))
    features["candsent_NER"] =  list(features.apply(lambda x: preprocess_NER(x["candsenttag"]), axis=1))
    
    features["origsent_NER_unigram"] = features["origsent_NER"]
    features["candsent_NER_unigram"] = features["candsent_NER"]
    
    features["origsent_Event"] =  list(features.apply(lambda x: preprocess_Event(x["origsenttag"]), axis=1))
    features["candsent_Event"] =  list(features.apply(lambda x: preprocess_Event(x["candsenttag"]), axis=1))
    
    features["origsent_Event_unigram"] = features["origsent_Event"]
    features["candsent_Event_unigram"] = features["candsent_Event"]
    
    print "generate bigram for Tags"
    feattag = ["origsent_tag", "candsent_tag", "origsent_NER", "candsent_NER", "origsent_Event", "candsent_Event"]
    for feat in feattag:
        join_str = "_"
        features["%s_bigram"%(feat)] = list(features.apply(lambda x: getBigram(x["%s_unigram"%(feat)], join_str), axis=1))
        features["%s_trigram"%(feat)] = list(features.apply(lambda x: getTrigram(x["%s_unigram"%(feat)], join_str), axis=1))
    
    gram_tags = ["_tag_unigram", "_tag_bigram", "_tag_trigram"]
    
    for gram in gram_tags:
        features["common-words_%s"%(gram)] = list(features.apply(lambda x: len(intersect(x["origsent"+gram], x["candsent"+gram])) , axis=1))
        features["levenshtein_%s"%(gram)] = list(features.apply(lambda x: distance.nlevenshtein(x["origsent"+gram], x["candsent"+gram], method=2) , axis=1))
        features["sorensen_%s"%(gram)] = list(features.apply(lambda x: distance.sorensen(x["origsent"+gram], x["candsent"+gram]), axis=1))
        features["cosine_%s"%(gram)] = list(features.apply(lambda x: cosine(x["origsent"+gram], x["candsent"+gram]), axis=1))
        features["precision_%s"%(gram)] = list(features.apply(lambda x: precision_recall(x["origsent"+gram], x["candsent"+gram], x["origsent"+gram]), axis=1))
        features["recall1gram_%s"%(gram)] = list(features.apply(lambda x: precision_recall(x["origsent"+gram], x["candsent"+gram], x["candsent"+gram]), axis=1))
        features["f1gram_%s"%(gram)] = list(features.apply(lambda x: fmeasure(x["precision_%s"%(gram)], x["recall1gram_%s"%(gram)]), axis=1))
    
    features["common_Event"] = list(features.apply(lambda x: len(intersect(x["origsent_Event"], x["candsent_Event"])) , axis=1))

    features["common_NER"] = list(features.apply(lambda x: len(intersect(x["origsent_Event"], x["candsent_Event"])) , axis=1))

    return features

def pos_tag_features(features):
    
    return features
    
    
def ner_features(features):
    features["origsent_NER"] =  list(features.apply(lambda x: preprocess_NER(x["origsenttag"]), axis=1))
    features["candsent_NER"] =  list(features.apply(lambda x: preprocess_NER(x["candsenttag"]), axis=1))
    
    features["origsent_NER_unigram"] = features["origsent_NER"]
    features["candsent_NER_unigram"] = features["candsent_NER"]
    
    print "generate bigram for Tags"
    feattag = ["origsent_NER", "candsent_NER"]
    for feat in feattag:
        join_str = "_"
        features["%s_bigram"%(feat)] = list(features.apply(lambda x: getBigram(x["%s_unigram"%(feat)], join_str), axis=1))
        features["%s_trigram"%(feat)] = list(features.apply(lambda x: getTrigram(x["%s_unigram"%(feat)], join_str), axis=1))
    
    features["common_NER"] = list(features.apply(lambda x: len(intersect(x["origsent_Event"], x["candsent_Event"])) , axis=1))

    
def event_features(features):
    features["origsent_Event"] =  list(features.apply(lambda x: preprocess_Event(x["origsenttag"]), axis=1))
    features["candsent_Event"] =  list(features.apply(lambda x: preprocess_Event(x["candsenttag"]), axis=1))
    
    features["origsent_Event_unigram"] = features["origsent_Event"]
    features["candsent_Event_unigram"] = features["candsent_Event"]
    
    print "generate bigram for Tags"
    feattag = ["origsent_Event", "candsent_Event"]
    for feat in feattag:
        join_str = "_"
        features["%s_bigram"%(feat)] = list(features.apply(lambda x: getBigram(x["%s_unigram"%(feat)], join_str), axis=1))
        features["%s_trigram"%(feat)] = list(features.apply(lambda x: getTrigram(x["%s_unigram"%(feat)], join_str), axis=1))
    
    features["common_Event"] = list(features.apply(lambda x: len(intersect(x["origsent_Event"], x["candsent_Event"])) , axis=1))

    
    
# In[7]:

feature_extraction(train)
extract_count_feat(train)
extract_basic_distance_feat(train)
#pos_tag_features(train)
#ner_features(train)
#event_features(train)
#wordnet = pa.read_csv("train_word_net.csv", sep="\t")
#train["wordnet-similarity"] = wordnet["wordnet-similarity"]
#train["wordnet-similarity-norm"] = wordnet["wordnet-similarity-norm"]
Xtrain, tfv_query = vectorize(train)


# In[8]:

print "test"
feature_extraction(test)
extract_count_feat(test)
extract_basic_distance_feat(test)
pos_tag_features(train)
#ner_features(train)
#event_features(train)
#wordnet_test = pa.read_csv("test_word_net.csv", sep="\t")
#test["wordnet-similarity"] = wordnet_test["wordnet-similarity"]
#test["wordnet-similarity-norm"] = wordnet_test["wordnet-similarity-norm"]
Xtest, _ = vectorize(test, tfv_query)


# In[9]:

#xgb_model = XGBClassifier()  
#clf = GridSearchCV(xgb_model, 
#                   {'max_depth': [4], 
#                    'n_estimators': [100], #tried with 50, 500, 1000 as well but best parameters is 100
#                    'learning_rate': [0.01]}, verbose=1)
#clf.fit(Xtrain,ytrain)
#clf.best_score_
#clf.best_params_


# In[10]:

xgb_ = XGBClassifier(max_depth=4, learning_rate=0.01, n_estimators = 100).fit(Xtrain, ytrain)
ytest_pred = xgb_.predict(Xtest) 


# In[11]:

print f1_score(ytest, ytest_pred, average="binary")
print precision_score(ytest, ytest_pred, average="binary")
print recall_score(ytest, ytest_pred, average="binary")   

f1_score(ytest, ytest_pred, average="binary")
# In[12]:

print accuracy_score(ytest, ytest_pred, normalize=True)



xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 4}
num_rounds = 100

dtrain = xgb.DMatrix(Xtrain, label=ytrain)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)


importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pa.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')