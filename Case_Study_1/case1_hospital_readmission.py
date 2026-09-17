import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

files.upload()
df = pd.read_csv('hospital_readmissions_30k.csv').drop(columns='patient_id')
bp = df['blood_pressure'].str.split('/', expand=True).astype(float)
df['systolic_bp'], df['diastolic_bp'] = bp[0], bp[1]
df = df.drop(columns='blood_pressure')
X = df.drop(columns='readmitted_30_days')
y = df['readmitted_30_days'].map({'No':0, 'Yes':1})
cat = X.select_dtypes('object').columns
num = X.select_dtypes(exclude='object').columns
prep = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model = Pipeline([('prep',prep),('lr',LogisticRegression(penalty='l2',C=1,max_iter=2000,random_state=42))])
model.fit(Xtr,ytr)
prob = model.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte,prob)
print('ROC-AUC:',round(auc,4))
fpr,tpr,_ = roc_curve(yte,prob)
plt.plot(fpr,tpr,label=f'AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.show()

# Threshold/cost analysis: example only; replace costs with approved clinical costs.
fn_cost, fp_cost = 5, 1
rows=[]
for th in [i/100 for i in range(10,91)]:
    p=(prob>=th).astype(int); tn,fp,fn,tp=confusion_matrix(yte,p).ravel()
    rows.append([th,fp,fn,fn_cost*fn+fp_cost*fp])
costs=pd.DataFrame(rows,columns=['threshold','FP','FN','total_cost'])
print('\nThreshold analysis:')
print(costs[costs.threshold.isin([.3,.4,.5,.6])])
print('\nIllustrative minimum-cost threshold:',costs.loc[costs.total_cost.idxmin(),'threshold'])
