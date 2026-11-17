# Robustness analyses: label-shuffling control, learning curve under study-grouped CV,
# and leave-one-catalyst-class-out validation. Requires clean.pkl from analysis_pipeline.py.
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
d=pd.read_pickle('clean.pkl')
m=d[d['yield'].notna()].reset_index(drop=True)
num=['wl','temp','cat_mol','has_add','intra']+[f'{p}_{k}' for p in ['s1','s2','p'] for k in ['len','heavy','arom','stereo','dbl','ring']]
X=pd.concat([m[num],pd.get_dummies(m[['cat','solv']]).astype(int)],axis=1); X['cat_mol_missing']=m.cat_mol.isna().astype(int)
y=m['yield'].values; g=m.Study_ID.values
RF=lambda: make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=0,n_jobs=-1))
out={}
# 1. label shuffling under random CV
def rcv(yv,seed):
    p=cross_val_predict(RF(),X,yv,cv=KFold(5,shuffle=True,random_state=seed)); return r2_score(yv,p)
real=[rcv(y,s) for s in range(5)]
rng=np.random.RandomState(1); null=[rcv(rng.permutation(y),s) for s in range(20)]
out['shuffle']={'real_mean':float(np.mean(real)),'null_mean':float(np.mean(null)),'null_max':float(np.max(null)),'null_sd':float(np.std(null)),'n_perm':20,
               'p_value':float((1+sum(n>=np.mean(real) for n in null))/(1+len(null)))}
print('shuffle',out['shuffle'])
# 2. learning curve: hold out 20% of studies as test (repeated), train on growing fraction of remaining studies
def grouped_split(seed,frac_test=0.2):
    ug=np.unique(g); r=np.random.RandomState(seed); r.shuffle(ug)
    sizes=pd.Series(g).value_counts(); test=[];n=0
    for s in ug:
        if n>=frac_test*len(y): break
        test.append(s); n+=sizes[s]
    te=np.isin(g,test); return np.where(~te)[0],np.where(te)[0]
lc={}
for frac in [0.25,0.5,0.75,1.0]:
    gr,rr=[],[]
    for seed in range(20):
        tr,te=grouped_split(seed)
        r=np.random.RandomState(100+seed); trs=np.unique(g[tr]); r.shuffle(trs); keep=trs[:max(2,int(round(frac*len(trs))))]
        tr2=tr[np.isin(g[tr],keep)]
        mdl=RF().fit(X.iloc[tr2],y[tr2]); gr.append(r2_score(y[te],mdl.predict(X.iloc[te])))
        # random split of same size for comparison
        idx=np.random.RandomState(200+seed).permutation(len(y)); te_r=idx[:len(te)]; pool=idx[len(te):][:len(tr2)]
        mdl=RF().fit(X.iloc[pool],y[pool]); rr.append(r2_score(y[te_r],mdl.predict(X.iloc[te_r])))
    lc[frac]={'n_train_mean':None,'grouped_R2':float(np.mean(gr)),'grouped_sd':float(np.std(gr)),'random_R2':float(np.mean(rr)),'random_sd':float(np.std(rr))}
    print('LC',frac,lc[frac])
out['learning_curve']=lc
# 3. leave-one-catalyst-class-out (classes with >= 10 yield records)
lco={}
for c,cnt in m.cat.value_counts().items():
    if cnt<10: continue
    te=(m.cat==c).values
    mdl=RF().fit(X[~te],y[~te]); p=mdl.predict(X[te])
    lco[c]={'n':int(te.sum()),'MAE_pct':float(mean_absolute_error(y[te],p)*100),'MAE_meanpred_pct':float(mean_absolute_error(y[te],np.full(te.sum(),y[~te].mean()))*100),
            'R2':float(r2_score(y[te],p))}
    print('LCO',c,lco[c])
out['leave_class_out']=lco
json.dump(out,open('robustness.json','w'),indent=1)
