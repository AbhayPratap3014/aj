!pip -q install imbalanced-learn xgboost

import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

files.upload()
df=pd.read_csv('Creditcard_data.csv')
X=df.drop(columns='Class'); y=df.Class
Xtr0,Xte,ytr0,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
Xtr,Xval,ytr,yval=train_test_split(Xtr0,ytr0,test_size=.25,random_state=42,stratify=ytr0)

# SMOTE only on training data; validation/test remain untouched.
Xtr,ytr=SMOTE(random_state=42,k_neighbors=3).fit_resample(Xtr,ytr)
model=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=42,n_jobs=-1)
model.fit(Xtr,ytr)
vp=model.predict_proba(Xval)[:,1]; tp=model.predict_proba(Xte)[:,1]
print('Validation ROC-AUC:',round(roc_auc_score(yval,vp),4))
print('Validation PR-AUC:',round(average_precision_score(yval,vp),4))
print('Test ROC-AUC:',round(roc_auc_score(yte,tp),4))
print('Test PR-AUC:',round(average_precision_score(yte,tp),4))

# Tune threshold on validation set using F1.
rows=[]
for th in [i/100 for i in range(5,96)]:
    p=(vp>=th).astype(int); tn,fp,fn,tpr=confusion_matrix(yval,p).ravel()
    prec=tpr/(tpr+fp) if tpr+fp else 0; rec=tpr/(tpr+fn) if tpr+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    rows.append([th,prec,rec,f1])
th=pd.DataFrame(rows,columns=['threshold','precision','recall','f1'])
best=th.loc[th.f1.idxmax()]; threshold=float(best.threshold)
print('\nSelected threshold:',threshold); print(best)

pred=(tp>=threshold).astype(int)
print('\nConfusion matrix:\n',confusion_matrix(yte,pred))
print('\nClassification report:\n',classification_report(yte,pred,digits=4))

precision,recall,_=precision_recall_curve(yte,tp)
plt.plot(recall,precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Fraud Precision-Recall Curve'); plt.show()

imp=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False).head(15).sort_values()
imp.plot(kind='barh',figsize=(8,6)); plt.xlabel('Importance'); plt.title('Top XGBoost Feature Importances'); plt.show()
print(imp.sort_values(ascending=False))
