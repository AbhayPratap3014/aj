!pip -q install openpyxl scikit-learn

import io, zipfile, requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# UCI Online Retail dataset
url='https://archive.ics.uci.edu/static/public/352/online+retail.zip'
z=zipfile.ZipFile(io.BytesIO(requests.get(url).content)); name=[n for n in z.namelist() if n.endswith('.xlsx')][0]
with z.open(name) as f: df=pd.read_excel(f)
df=df.dropna(subset=['CustomerID']); df=df[df.Quantity>0]; df['InvoiceDate']=pd.to_datetime(df.InvoiceDate); df['Revenue']=df.Quantity*df.UnitPrice
snap=df.InvoiceDate.max()+pd.Timedelta(days=1)
rfm=df.groupby('CustomerID').agg(Recency=('InvoiceDate',lambda x:(snap-x.max()).days),Frequency=('InvoiceNo','nunique'),Monetary=('Revenue','sum')).dropna()

# Log transform reduces skew; standardize before clustering.
Z=StandardScaler().fit_transform(np.log1p(rfm[['Recency','Frequency','Monetary']]))
k=KMeans(n_clusters=4,n_init=10,random_state=42); rfm['KMeans']=k.fit_predict(Z)
print('KMeans silhouette:',round(silhouette_score(Z,rfm.KMeans),4))

db=DBSCAN(eps=.8,min_samples=8); rfm['DBSCAN']=db.fit_predict(Z); mask=rfm.DBSCAN!=-1
if mask.sum()>10 and rfm.loc[mask,'DBSCAN'].nunique()>1: print('DBSCAN silhouette:',round(silhouette_score(Z[mask],rfm.loc[mask,'DBSCAN']),4))
print('DBSCAN counts:\n',rfm.DBSCAN.value_counts().sort_index())

pca=PCA(n_components=2,random_state=42); p=pca.fit_transform(Z)
plt.scatter(p[:,0],p[:,1],c=rfm.KMeans,s=12); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('K-Means Segments'); plt.show()

n=min(2000,len(rfm)); idx=np.random.RandomState(42).choice(len(rfm),n,replace=False)
t=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=42).fit_transform(Z[idx])
plt.scatter(t[:,0],t[:,1],c=rfm.iloc[idx].KMeans,s=12); plt.title('t-SNE Customer Segments'); plt.show()

profiles=rfm.groupby('KMeans')[['Recency','Frequency','Monetary']].mean().round(2)
med=rfm[['Recency','Frequency','Monetary']].median()
def persona(r):
    if r.Frequency>=med.Frequency and r.Monetary>=med.Monetary: return 'High-value loyal'
    if r.Recency<=med.Recency and r.Monetary<med.Monetary: return 'Recent low-value'
    if r.Recency>med.Recency and r.Frequency<med.Frequency: return 'At-risk / inactive'
    return 'Occasional buyer'
profiles['Persona']=profiles.apply(persona,axis=1)
print('\nRFM profiles and personas:\n',profiles)
