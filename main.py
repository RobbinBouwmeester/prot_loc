from collections import Counter
import os
import random
import itertools

import operator

import pickle

import pandas as pd
import matplotlib.pyplot as plt

#SciPy
import scipy.stats as st
import scipy

#ML
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import numpy
import numpy as np

import copy
from scipy.stats import randint
from scipy.stats import uniform
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#from keras.layers import Dropout

from train_model import train_xgb
from feat_extract import get_feats

def get_preds(main_path="",infile="data/test.mfa",model="",outfile_preds="preds_mod_v2.csv"):
    data_df = get_feats(infile_name=os.path.join(main_path,infile))
    # TODO if we do not define the class, exclude it as a feature
    data_df.pop("class")
    mod = pickle.load(open(os.path.join(main_path,model),"rb"))
    prob_pos = pd.Series(mod.predict_proba(data_df)[:,1],index=data_df.index,name="predictions")
    prob_pos.to_csv(outfile_preds)

def train_all_instances(main_path="",
                        input_depleted="data/train_depleted.csv",
                        input_enriched="data/train_enriched.csv",
                        random_seed=42,
                        output_file_mod="mods/model.pickle"):
    random.seed(random_seed)
    
    data_df_one = get_feats(infile_name=os.path.join(main_path,input_enriched),assign_class=1)
    data_df_zero = get_feats(infile_name=os.path.join(main_path,input_depleted),assign_class=0)
    
    data_df = []
    data_df.append(data_df_one)
    data_df.append(data_df_zero)
    data_df = pd.concat(data_df)

    y = data_df.pop("class")

    train_xgb(data_df,y,outfile=output_file_mod)

if __name__ == "__main__":
    main_path="C:/Users/asus/Documents/GitHub/prot_loc"
    output_file_mod="mods/modelV3.pickle"
    train_all_instances(main_path=main_path,input_depleted="data/zero.mfa",input_enriched="data/one.mfa",random_seed=42,output_file_mod=output_file_mod)
    get_preds(main_path=main_path,infile="data/test.mfa",model=output_file_mod,outfile_preds="preds_mod_v3.csv")
