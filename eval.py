from sklearn.metrics import matthews_corrcoef
import pandas as pd

infile_preds = "preds_mod_v1.csv"
preds = pd.read_csv(infile_preds,index_col=0)
actual = pd.read_csv("data/test.csv",index_col=0)

preds["predictions"][preds["predictions"] > 0.5] = 1.0
preds["predictions"][preds["predictions"] < 0.51] = 0.0

all_df = pd.concat([preds,actual],axis=1)

print("MCC for %s: %.3f" % (infile_preds,matthews_corrcoef(all_df["predictions"],all_df["target"])))

infile_preds = "preds_mod_v2.csv"
preds = pd.read_csv(infile_preds,index_col=0)
actual = pd.read_csv("data/test.csv",index_col=0)

preds["predictions"][preds["predictions"] > 0.5] = 1.0
preds["predictions"][preds["predictions"] < 0.51] = 0.0

all_df = pd.concat([preds,actual],axis=1)

print("MCC for %s: %.3f" % (infile_preds,matthews_corrcoef(all_df["predictions"],all_df["target"])))