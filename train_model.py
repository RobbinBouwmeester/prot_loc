import pickle

import scipy.stats as st

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef,make_scorer

def train_xgb(X,
              y,
              mod_number=1,
              cv=None,
              outfile="model.pickle",
              n_iter_search=20,
              nfolds=10,
              random_state=42):
    """
    Train an XGBoost model with hyper parameter optimization.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
        
    Returns
    -------
    object
        Trained XGBoost model
    object
        Cross-validation results
    """
    
    xgb_handle = xgb.XGBClassifier()

    one_to_left = st.beta(10, 1)  
    from_zero_positive = st.expon(0, 50)
    
    #Define distributions to sample from for hyper parameter optimization
    param_dist = {  
        "n_estimators": st.randint(5, 150),
        "max_depth": st.randint(5, 10),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    if not cv: cv = KFold(n_splits=nfolds, shuffle=True,random_state=random_state)

    mcc = make_scorer(matthews_corrcoef)
    random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=0,scoring=mcc,
                                       n_jobs=1,refit=True,cv=cv)

    random_search.fit(X, y)

    random_search.feats = X.columns
    pickle.dump(random_search,open(outfile,"wb"))

    return(random_search.best_score_)