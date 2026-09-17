!pip -q install aif360 shap

import numpy as np
import pandas as pd
import shap
from aif360.datasets import GermanDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# IBM AIF360 German Credit dataset; protected attribute used below is sex.
data=GermanDataset(protected_attribute_names=['sex'],privileged_classes=[[1]],favorable_classes=[1])
train,test=data.split([.7],shuffle=True,seed=42)

# ---------------- Baseline ----------------
base=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(max_iter=2000,random_state=42))])
base.fit(train.features,train.labels.ravel())
train_pred=train.copy(); train_pred.labels=base.predict(train.features).reshape(-1,1)
test_pred=test.copy(); test_pred.labels=base.predict(test.features).reshape(-1,1)

def fairness(true,pred):
    m=ClassificationMetric(true,pred,unprivileged_groups={'sex':0},privileged_groups={'sex':1})
    return m.statistical_parity_difference(),m.equalized_odds_difference()
print('Baseline demographic parity difference, equalized odds difference:',fairness(test,test_pred))

# ---------------- Reweighing ----------------
rw=Reweighing(unprivileged_groups={'sex':0},privileged_groups={'sex':1})
train_rw=rw.fit_transform(train)
weighted=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(max_iter=2000,random_state=42))])
weighted.fit(train_rw.features,train_rw.labels.ravel(),lr__sample_weight=train_rw.instance_weights.ravel())
rw_pred=test.copy(); rw_pred.labels=weighted.predict(test.features).reshape(-1,1)
print('Reweighed demographic parity difference, equalized odds difference:',fairness(test,rw_pred))

# ---------------- Equalized-odds post-processing ----------------
eq=EqOddsPostprocessing(unprivileged_groups={'sex':0},privileged_groups={'sex':1},seed=42)
eq.fit(train,train_pred)
eq_pred=eq.predict(test_pred)
print('Post-processed demographic parity difference, equalized odds difference:',fairness(test,eq_pred))

# ---------------- SHAP individual explanation ----------------
scaler=base.named_steps['scale']; lr=base.named_steps['lr']
explainer=shap.LinearExplainer(lr,scaler.transform(train.features))
sv=explainer(scaler.transform(test.features))
i=0
print('Individual predicted probability:',base.predict_proba(test.features[i:i+1])[0,1])
shap.plots.waterfall(sv[i],max_display=12)
