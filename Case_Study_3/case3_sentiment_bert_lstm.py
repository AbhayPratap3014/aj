!pip -q install transformers datasets accelerate

import re, numpy as np, torch
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Amazon polarity: 0=negative, 1=positive. Small subsets keep Colab runtime practical.
ds=load_dataset('amazon_polarity')
train_ds=ds['train'].shuffle(seed=42).select(range(8000))
test_ds=ds['test'].shuffle(seed=42).select(range(2000))

# ---------------- BERT ----------------
tok=AutoTokenizer.from_pretrained('bert-base-uncased')
def enc(x): return tok(x['title'],x['content'],truncation=True,padding='max_length',max_length=128)
tr=train_ds.map(enc,batched=True).remove_columns(['title','content']); te=test_ds.map(enc,batched=True).remove_columns(['title','content'])
tr.set_format('torch'); te.set_format('torch')
bert=AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
def metrics(p): return {'accuracy':accuracy_score(p.label_ids,np.argmax(p.predictions,axis=1))}
args=TrainingArguments(output_dir='./bert_out',eval_strategy='epoch',save_strategy='no',learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=8,num_train_epochs=1,weight_decay=.01,report_to='none')
trainer=Trainer(model=bert,args=args,train_dataset=tr,eval_dataset=te,compute_metrics=metrics)
trainer.train(); print('BERT:',trainer.evaluate())

# ---------------- BiLSTM baseline ----------------
def words(s): return re.findall(r"\b[a-zA-Z']+\b",s.lower())
texts=[(x['title']+' '+x['content']).strip() for x in train_ds]
test_texts=[(x['title']+' '+x['content']).strip() for x in test_ds]
c=Counter(); [c.update(words(x)) for x in texts]
vocab={'<PAD>':0,'<UNK>':1}
for w,_ in c.most_common(20000): vocab[w]=len(vocab)
L=120
def encode(s):
    a=[vocab.get(w,1) for w in words(s)[:L]]; return a+[0]*(L-len(a))
class Reviews(Dataset):
    def __init__(self,texts,labels): self.x=torch.tensor([encode(s) for s in texts]); self.y=torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]
tr_loader=DataLoader(Reviews(texts,train_ds['label']),batch_size=64,shuffle=True)
te_loader=DataLoader(Reviews(test_texts,test_ds['label']),batch_size=64)
class BiLSTM(nn.Module):
    def __init__(self,n):
        super().__init__(); self.e=nn.Embedding(n,128,padding_idx=0); self.l=nn.LSTM(128,128,batch_first=True,bidirectional=True); self.fc=nn.Linear(256,2)
    def forward(self,x): return self.fc(self.l(self.e(x))[0][:,-1])
device='cuda' if torch.cuda.is_available() else 'cpu'; net=BiLSTM(len(vocab)).to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-3); loss=nn.CrossEntropyLoss()
for ep in range(2):
    net.train()
    for x,y in tr_loader:
        x,y=x.to(device),y.to(device); opt.zero_grad(); loss(net(x),y).backward(); opt.step()
    print('LSTM epoch',ep+1,'done')
net.eval(); pred=[]; actual=[]
with torch.no_grad():
    for x,y in te_loader: pred.extend(net(x.to(device)).argmax(1).cpu().numpy()); actual.extend(y.numpy())
print('LSTM accuracy:',accuracy_score(actual,pred)); print(classification_report(actual,pred))
