"""Analysis pipeline for the photochemical [2+2] dataset.
Place ML_merged_cleaned.xlsx in the working directory and run: python analysis_pipeline.py
Requires pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl. Writes clean.pkl, cv.json, figures fig1-3.png and
ML_cleaned_dataset.csv.
"""

# ==================== analysis.py ====================
import pandas as pd, numpy as np, re, json, warnings, datetime
warnings.filterwarnings('ignore')
d=pd.read_excel('ML_merged_cleaned.xlsx')
# ---- apply documented corrections (Excel drag-fill artifacts) and attach traced provenance
import os
def _find(f):
    for p in (f, os.path.join('data', f), os.path.join('..', 'data', f)):
        if os.path.exists(p): return p
    raise FileNotFoundError(f)
_C=pd.read_csv(_find('corrections.csv'),dtype=str)
for _,c in _C.iterrows():
    if c.Action=='corrected':
        d.loc[d.Reaction_ID==int(c.Reaction_ID),c.Field]=c.Corrected
    elif c.Action=='set to missing':
        d.loc[d.Reaction_ID==int(c.Reaction_ID),c.Field]=np.nan
_P=pd.read_csv(_find('provenance.csv'))
d=d.merge(_P[['Reaction_ID','Study_ID','Review_scheme','Primary_DOIs','Confidence']],on='Reaction_ID',how='left')
def td(v):
    if isinstance(v,datetime.timedelta):
        m=int(round(v.total_seconds()/60)); return m//60,m%60
    if isinstance(v,str):
        mm=re.match(r'^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$',v)
        if mm: return float(mm.group(1)),float(mm.group(2))   # ratios Excel left as text, e.g. '97.5:2.5'
    return (np.nan,np.nan)
for c,n in [('Diastereomeric_ratio','dr'),('Enantiomeric_ratio','er')]:
    ab=d[c].map(td); d[n+'_a']=[x[0] for x in ab]; d[n+'_b']=[x[1] for x in ab]
    d[n+'_major']=d[n+'_a']/(d[n+'_a']+d[n+'_b'])
d['yield']=pd.to_numeric(d['Yield'],errors='coerce')
d['time_h']=pd.to_numeric(d['Time_hr'].astype(str).str.extract(r'([\d.]+)')[0],errors='coerce')
def wl(s):
    s=str(s); n=[float(x) for x in re.findall(r'\d+',s)]
    if not n: return np.nan
    return np.mean(n) if len(n)>1 else n[0]
d['wl']=d.Light_nm.map(wl)
d['temp']=d['Temp_°C'].map(lambda x:25.0 if str(x).strip().lower() in('r.t.','rt') else pd.to_numeric(x,errors='coerce'))
d['intra']=d.Substrate_2_SMILES.isna().astype(int)
def cat(s):
    if pd.isna(s): return 'none'
    if 'Ru' in s: return 'Ru'
    if 'Cu' in s: return 'Cu'
    if 'Al' in s or 'B' in s and 'N+' in s: return 'LewisAcid'
    if '[O+]' in s or 'o+' in s: return 'pyrylium'
    return 'other_org'
d['cat']=d.Catalyst_SMILES.map(cat)
def solv(s):
    if pd.isna(s): return 'unk'
    m={'CC#N':'MeCN','N#CC':'MeCN','ClCCl':'DCM','CCOCC':'Et2O','CC1=CC=CC=C1':'toluene','C1=CC=CC=C1':'benzene','CC(C)=O':'acetone','CCCCCC':'hexane','C1CCCCC1':'hexane'}
    return m.get(s,'other')
d['solv']=d.Solvent.map(solv)
def cnt(s,pat): return 0 if pd.isna(s) else len(re.findall(pat,s))
for nm,col in [('s1','Substrate_1_SMILES'),('s2','Substrate_2_SMILES'),('p','Product_SMILES')]:
    d[nm+'_len']=d[col].fillna('').str.len()
    d[nm+'_heavy']=d[col].map(lambda s:cnt(s,r'Cl|Br|Si|[CNOSFPBI]|c|n|o|s'))
    d[nm+'_arom']=d[col].map(lambda s:cnt(s,r'c|n(?![a-z])|o(?![a-z])'))
    d[nm+'_stereo']=d[col].map(lambda s:cnt(s,'@'))
    d[nm+'_dbl']=d[col].map(lambda s:cnt(s,'='))
    d[nm+'_ring']=d[col].map(lambda s:cnt(s,r'\d')) 
d['cat_mol']=d['Cat.(mol%)']
d['has_add']=d.Reagent_Second.notna().astype(int)
d.to_pickle('clean.pkl')
out={}
out['n']=len(d); out['yield_n']=int(d['yield'].notna().sum())
out['yield_desc']=d['yield'].describe().to_dict()
out['time_n']=int(d.time_h.notna().sum()); out['dr_n']=int(d.dr_major.notna().sum()); out['er_n']=int(d.er_major.notna().sum())
out['dr_desc']=d.dr_major.describe().to_dict()
out['intra']=int(d.intra.sum())
out['cat']=d.cat.value_counts().to_dict(); out['solv']=d.solv.value_counts().to_dict()
out['nsrc']=int(d.Source_DOI.nunique()); out['nstudy']=int(d.Study_ID.nunique()); out['study_sizes']=d.Study_ID.value_counts().describe().to_dict(); out['srcmax']=d.Source_DOI.value_counts().head(5).to_dict()
out['uv']=int((d.wl<400).sum()); out['vis']=int((d.wl>=400).sum())
out['tempd']=d.temp.describe().to_dict(); out['rt']=int((d.temp==25).sum()); out['subzero']=int((d.temp<0).sum())
out['yield_by_cat']=d.groupby('cat')['yield'].agg(['count','mean','std']).round(3).to_dict('index')
out['yield_by_intra']=d.groupby('intra')['yield'].agg(['count','mean','std']).round(3).to_dict('index')
out['uniq']={k:int(d[k].nunique()) for k in ['Substrate_1_SMILES','Substrate_2_SMILES','Catalyst_SMILES','Product_SMILES']}
out['wl_desc']=d.wl.describe().to_dict()
out['missing']={c:int(d[c].isna().sum()) for c in ['Substrate_2_SMILES','Catalyst_SMILES','Cat.(mol%)','Solvent','Time_hr','Yield','Diastereomeric_ratio','Enantiomeric_ratio','Major_Product_Syn_%','Reagent_Second']}
print(json.dumps(out,indent=1,default=float))

# ==================== model.py ====================
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
d=pd.read_pickle('clean.pkl')
m=d[d['yield'].notna()].reset_index(drop=True)
num=['wl','temp','cat_mol','has_add','intra']+[f'{p}_{k}' for p in ['s1','s2','p'] for k in ['len','heavy','arom','stereo','dbl','ring']]
X=pd.concat([m[num],pd.get_dummies(m[['cat','solv']]).astype(int)],axis=1)
X['cat_mol_missing']=m.cat_mol.isna().astype(int)
y=m['yield'].values; g=m.Study_ID.values
print(X.shape)
models={'Mean baseline':DummyRegressor(),
 'Ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-1,3,20))),
 'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=0,n_jobs=-1)),
 'Gradient Boosting':make_pipeline(SimpleImputer(strategy='median'),GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0))}
res={}
def ev(name,mdl,scheme):
    r2s,maes,rms=[],[],[]
    if scheme=='random':
        for rep in range(5):
            kf=KFold(5,shuffle=True,random_state=rep); p=cross_val_predict(mdl,X,y,cv=kf)
            r2s.append(r2_score(y,p));maes.append(mean_absolute_error(y,p));rms.append(mean_squared_error(y,p)**.5)
    else:
        # grouped: shuffle group assignment across repeats
        ug=np.unique(g)
        for rep in range(5):
            rng=np.random.RandomState(rep); perm=rng.permutation(len(ug)); mp={u:perm[i]%5 for i,u in enumerate(ug)}
            # balance: assign biggest groups first to different folds
            sizes=pd.Series(g).value_counts(); folds=[[] for _ in range(5)]; load=[0]*5
            order=list(sizes.index); 
            big=order[:5]; rng.shuffle(big); rest=order[5:]; rng.shuffle(rest)
            for gg in big+rest:
                k=int(np.argmin(load)); folds[k].append(gg); load[k]+=sizes[gg]
            p=np.zeros(len(y))
            for k in range(5):
                te=np.isin(g,folds[k]); tr=~te
                mdl.fit(X[tr],y[tr]); p[te]=mdl.predict(X[te])
            r2s.append(r2_score(y,p));maes.append(mean_absolute_error(y,p));rms.append(mean_squared_error(y,p)**.5)
    return [np.mean(r2s),np.std(r2s),np.mean(maes),np.mean(rms)]
for n,mdl in models.items():
    for s in ['random','grouped']:
        res[(n,s)]=ev(n,mdl,s); print(n,s,np.round(res[(n,s)],3))
json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open('cv.json','w'))
# hold-out 80/20 RF
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
rf=models['Random Forest'].fit(Xtr,ytr); pt=rf.predict(Xte)
print('holdout RF',r2_score(yte,pt),mean_absolute_error(yte,pt),mean_squared_error(yte,pt)**.5,len(yte))
json.dump({'r2':r2_score(yte,pt),'mae':mean_absolute_error(yte,pt),'rmse':mean_squared_error(yte,pt)**.5,'n_te':len(yte),'n_tr':len(ytr)},open('holdout.json','w'))
np.save('hold.npy',np.c_[yte,pt])
# permutation importance under grouped CV predictions? use full-data RF w/ permutation on OOF-free: use repeated holdout
imp=np.zeros(X.shape[1]); R=20
for r in range(R):
    Xa,Xb,ya,yb=train_test_split(X,y,test_size=.25,random_state=r)
    mm=models['Random Forest'].fit(Xa,ya)
    pi=permutation_importance(mm,Xb,yb,n_repeats=5,random_state=r,scoring='r2')
    imp+=pi.importances_mean
imp/=R
s=pd.Series(imp,index=X.columns).sort_values(ascending=False); print(s.head(12)); s.to_csv('imp.csv')
# grouped oof predictions for parity plot
ug=pd.Series(g).value_counts(); folds=[[] for _ in range(5)]; load=[0]*5
for gg in ug.index:
    k=int(np.argmin(load)); folds[k].append(gg); load[k]+=ug[gg]
p=np.zeros(len(y)); mdl=models['Random Forest']
for k in range(5):
    te=np.isin(g,folds[k]); mdl.fit(X[~te],y[~te]); p[te]=mdl.predict(X[te])
np.save('oof_grouped.npy',np.c_[y,p]); print('grouped oof r2',r2_score(y,p))
# dr descriptive by cat
dd=d[d.dr_major.notna()]; print(dd.groupby('cat').dr_major.agg(['count','mean']))
print(dd.groupby('intra').dr_major.agg(['count','mean']))
print(len(dd), dd.Source_DOI.nunique())

# ==================== stats2.py ====================
import pandas as pd, numpy as np, json
from scipy import stats
d=pd.read_pickle('clean.pkl')
y=d[d['yield'].notna()]
a=y[y.intra==1]['yield'];b=y[y.intra==0]['yield']
print('intra vs inter',len(a),a.mean(),len(b),b.mean(),stats.mannwhitneyu(a,b))
uv=y[y.wl<400]['yield'];vi=y[y.wl>=400]['yield']; print('uv/vis',len(uv),uv.mean(),len(vi),vi.mean(),stats.mannwhitneyu(uv,vi))
la=y[y.cat=='LewisAcid']['yield'];rest=y[y.cat!='LewisAcid']['yield'];print('LA',la.mean(),rest.mean(),stats.mannwhitneyu(la,rest))
print('kruskal cat',stats.kruskal(*[g['yield'].values for _,g in y.groupby('cat')]))
# temp vs yield overall
t=y.dropna(subset=['temp']);print('temp-yield spearman',stats.spearmanr(t.temp,t['yield']))
# anie source
s=d[d.Source_DOI=='10.1002/anie.202001634']; print(len(s), s['yield'].describe(), s.temp.value_counts().to_dict(), s.wl.value_counts().to_dict(), s.Solvent.unique())
sd=s.dropna(subset=['dr_major','temp']); print(len(sd),stats.spearmanr(sd.temp,sd.dr_major))
print(s[['temp','yield','dr_major','er_major']].head(15))
# dr vs temp overall
dd=d.dropna(subset=['dr_major','temp']); print('dr-temp all',len(dd),stats.spearmanr(dd.temp,dd.dr_major))
print('dr sub0 vs rt', dd[dd.temp<0].dr_major.mean(), dd[dd.temp>=0].dr_major.mean(), (dd.temp<0).sum())
print('dr>=0.9',(d.dr_major>=.9).sum(), d.dr_major.notna().sum())
# syn%
sy=d['Major_Product_Syn_%'].dropna(); sy=sy.where(sy<=1,sy/100); print('syn',len(sy),sy.describe())
# time
print(d.time_h.describe())
# yield high
print('yield>=0.9',(y['yield']>=.9).sum(),'<0.5',(y['yield']<.5).sum(), len(y))
print('rows w/ catalyst+molpct',d[d.Catalyst_SMILES.notna()&d.cat_mol.notna()].shape[0])
print('cat none & UV',((d.cat=='none')&(d.wl<400)).sum(), 'cat none',(d.cat=='none').sum())
print('cat by uv/vis',pd.crosstab(d.cat,d.wl<400))
print('subzero',(d.temp<0).sum(),'temp lt -70',(d.temp<=-70).sum())
print('substrate2 unique',d.Substrate_2_SMILES.nunique(), d.Substrate_2_SMILES.value_counts().head(3).to_dict())

# smiles bracket check
def bal(s): return s.count('(')==s.count(')') and s.count('[')==s.count(']')
bad=[c for c in ['Substrate_1_SMILES','Substrate_2_SMILES','Catalyst_SMILES','Product_SMILES'] for s in d[c].dropna() if not bal(s)]; print('unbalanced',len(bad))
print('id unique',d.Reaction_ID.is_unique, d.Reaction_ID.min(), d.Reaction_ID.max())

# ==================== figs.py ====================
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
d=pd.read_pickle('clean.pkl'); y=d[d['yield'].notna()]
C='#1F4E79'; C2='#C55A11'
# Fig1
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
ax[0].hist(y['yield']*100,bins=np.arange(0,105,5),color=C,edgecolor='white')
ax[0].set_xlabel('Reported yield (%)');ax[0].set_ylabel('Number of reactions');ax[0].set_title('a  Yield distribution (n = 239)',loc='left',fontsize=9,fontweight='bold')
cols=['Yield','Substrate_2_SMILES','Catalyst_SMILES','Cat.(mol%)','Diastereomeric_ratio','Enantiomeric_ratio','Time_hr']
lab=['Yield','Substrate 2','Catalyst','Catalyst mol%','Diastereo. ratio','Enantio. ratio','Time']
avail=[100*d[c].notna().mean() for c in cols]
o=np.argsort(avail)
ax[1].barh(np.array(lab)[o],np.array(avail)[o],color=C2);ax[1].set_xlim(0,100);ax[1].set_xlabel('Records populated (%)')
ax[1].set_title('b  Field completeness (n = 249)',loc='left',fontsize=9,fontweight='bold')
for i,v in enumerate(np.array(avail)[o]): ax[1].text(v+1,i,f'{v:.0f}',va='center',fontsize=8)
plt.tight_layout();plt.savefig('fig1.png',dpi=200);plt.close()
# Fig2
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
order=['none','Ru','Cu','LewisAcid','pyrylium','other_org'];names=['None\n(direct)','Ru\npolypyridyl','Cu(I)','Al/B Lewis\nacid','Pyrylium','Other\norganic']
data=[y[y.cat==c]['yield']*100 for c in order]
bp=ax[0].boxplot(data,patch_artist=True,widths=.6,medianprops=dict(color='black'))
for b in bp['boxes']: b.set_facecolor('#9DC3E6')
ax[0].set_xticklabels([f'{n}\n(n={len(x)})' for n,x in zip(names,data)],fontsize=6.5);ax[0].set_ylabel('Yield (%)');ax[0].set_title('a  Yield by catalyst class',loc='left',fontsize=9,fontweight='bold')
data2=[y[y.intra==1]['yield']*100,y[y.intra==0]['yield']*100]
bp=ax[1].boxplot(data2,patch_artist=True,widths=.5,medianprops=dict(color='black'))
for b,c in zip(bp['boxes'],['#9DC3E6','#F4B183']): b.set_facecolor(c)
ax[1].set_xticklabels([f'Intramolecular\n(n={len(data2[0])})',f'Intermolecular\n(n={len(data2[1])})']);ax[1].set_title('b  Yield by reaction mode',loc='left',fontsize=9,fontweight='bold')
plt.tight_layout();plt.savefig('fig2.png',dpi=200);plt.close()
# Fig3
from sklearn.metrics import r2_score
h=np.load('hold.npy');g=np.load('oof_grouped.npy');imp=pd.read_csv('imp.csv',index_col=0).iloc[:,0]
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
ax[0].scatter(g[:,0]*100,g[:,1]*100,s=12,alpha=.6,color=C,label=f'Study-grouped CV (R² = {r2_score(g[:,0],g[:,1]):.2f})')
ax[0].plot([0,100],[0,100],'k--',lw=.8);ax[0].set_xlim(0,105);ax[0].set_ylim(0,105)
ax[0].set_xlabel('Reported yield (%)');ax[0].set_ylabel('Predicted yield (%)');ax[0].legend(frameon=False,fontsize=7,loc='upper left');ax[0].set_title('a  Parity plot (Random Forest)',loc='left',fontsize=9,fontweight='bold')
nm={'s2_len':'Substrate 2 SMILES length','temp':'Temperature','solv_MeCN':'Solvent = MeCN','p_stereo':'Product stereo-marks','s1_stereo':'Substrate 1 stereo-marks','p_ring':'Product ring digits','s2_stereo':'Substrate 2 stereo-marks','s1_heavy':'Substrate 1 heavy atoms','solv_toluene':'Solvent = toluene','p_heavy':'Product heavy atoms'}
t=imp.head(8)[::-1]
ax[1].barh([nm.get(i,i) for i in t.index],t.values,color=C2);ax[1].set_xlabel('Mean drop in R² when permuted');ax[1].tick_params(axis='y',labelsize=7)
ax[1].set_title('b  Permutation importance',loc='left',fontsize=9,fontweight='bold')
plt.tight_layout();plt.savefig('fig3.png',dpi=200);plt.close()

# ==================== export ====================
cols=['Reaction_ID','Substrate_1_SMILES','Substrate_2_SMILES','Catalyst_SMILES','Product_SMILES','Source_DOI','Review_scheme','Primary_DOIs','Study_ID','Confidence','Solvent','solv','cat','cat_mol','wl','temp','time_h','intra','yield','dr_a','dr_b','dr_major','er_a','er_b','er_major']
out=d[cols].rename(columns={'Review_scheme':'Review_scheme','Confidence':'Provenance_confidence','solv':'Solvent_class','cat':'Catalyst_class','cat_mol':'Cat_mol_pct','wl':'Wavelength_nm','temp':'Temp_C','time_h':'Time_h','intra':'Intramolecular','yield':'Yield_fraction','dr_a':'dr_major_part','dr_b':'dr_minor_part','dr_major':'dr_major_fraction','er_a':'er_major_part','er_b':'er_minor_part','er_major':'er_major_fraction'})
out.to_csv('ML_cleaned_dataset.csv',index=False)

# ==================== selectivity.py: classifier for major-isomer fraction >= 0.90 ====================
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
d=pd.read_pickle('clean.pkl')
m=d[d.dr_major.notna()].reset_index(drop=True)
num=['wl','temp','cat_mol','has_add','intra']+[f'{p}_{k}' for p in ['s1','s2','p'] for k in ['len','heavy','arom','stereo','dbl','ring']]
X=pd.concat([m[num],pd.get_dummies(m[['cat','solv']]).astype(int)],axis=1); X['cat_mol_missing']=m.cat_mol.isna().astype(int)
X=X.loc[:,X.nunique()>1]
y=(m.dr_major>=0.90).astype(int).values; g=m.Study_ID.values
print('n',len(y),'pos',y.sum(),'groups',len(set(g)),'features',X.shape[1])
print(pd.Series(g).value_counts().to_dict())
print(pd.crosstab(m.Study_ID,y).sum())
models={'Majority baseline':DummyClassifier(strategy='prior'),
 'Logistic regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=5000)),
 'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=0,n_jobs=-1))}
res={}
for n,mdl in models.items():
    # random stratified 5-fold x 10
    auc,bac,mcc=[],[],[]
    for r in range(10):
        p=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,y):
            mdl.fit(X.iloc[tr],y[tr]); p[te]=mdl.predict_proba(X.iloc[te])[:,1]
        auc.append(roc_auc_score(y,p)); yh=(p>=0.5).astype(int); bac.append(balanced_accuracy_score(y,yh)); mcc.append(matthews_corrcoef(y,yh))
    res[(n,'random')]=[np.mean(auc),np.std(auc),np.mean(bac),np.mean(mcc)]
    # leave-one-source-out
    p=np.zeros(len(y))
    for tr,te in LeaveOneGroupOut().split(X,y,g):
        if len(set(y[tr]))<2: p[te]=y[tr].mean(); continue
        mdl.fit(X.iloc[tr],y[tr]); p[te]=mdl.predict_proba(X.iloc[te])[:,1]
    yh=(p>=0.5).astype(int)
    res[(n,'LOSO')]=[roc_auc_score(y,p),0.0,balanced_accuracy_score(y,yh),matthews_corrcoef(y,yh)]
    if n=='Random Forest': np.save('sel_loso.npy',np.c_[y,p])
for k,v in res.items(): print(k,np.round(v,3))
json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open('sel.json','w'))

# ==================== fold assignments for reuse ====================
d=pd.read_pickle('clean.pkl')
fa=d[['Reaction_ID','Study_ID']].copy()
ym=d['yield'].notna().values
gy=d.loc[ym,'Study_ID'].values
for rep in range(5):
    rng=np.random.RandomState(rep); sizes=pd.Series(gy).value_counts(); order=list(sizes.index)
    big=order[:5]; rng.shuffle(big); rest=order[5:]; rng.shuffle(rest)
    fl=[[] for _ in range(5)]; load=[0]*5
    for gg in big+rest:
        k=int(np.argmin(load)); fl[k].append(gg); load[k]+=sizes[gg]
    col=np.full(len(d),np.nan)
    for k,f in enumerate(fl): col[ym & d.Study_ID.isin(f).values]=k
    fa[f'yield_grouped_fold_rep{rep}']=col
sm=d.dr_major.notna().values
srcs=sorted(d.loc[sm,'Study_ID'].unique())
fa['selectivity_loso_fold']=[srcs.index(s) if m else np.nan for s,m in zip(d.Study_ID,sm)]
fa.to_csv('fold_assignments.csv',index=False)
print('fold assignments written')

# ==================== bootstrap CIs for held-out-study selectivity AUC ====================
from sklearn.model_selection import LeaveOneGroupOut as _LOGO
_ci={}
for _nm in ['Logistic regression','Random Forest']:
    _mdl=models[_nm]; _p=np.zeros(len(y))
    for _tr,_te in _LOGO().split(X,y,g):
        if len(set(y[_tr]))<2: _p[_te]=y[_tr].mean(); continue
        _mdl.fit(X.iloc[_tr],y[_tr]); _p[_te]=_mdl.predict_proba(X.iloc[_te])[:,1]
    _rng=np.random.RandomState(0); _b=[]
    for _ in range(2000):
        _i=_rng.randint(0,len(y),len(y))
        if len(set(y[_i]))==2: _b.append(roc_auc_score(y[_i],_p[_i]))
    _ci[_nm]=[roc_auc_score(y,_p)]+list(np.percentile(_b,[2.5,97.5]))
    print('LOSO CI',_nm,np.round(_ci[_nm],3))
json.dump(_ci,open('sel_ci.json','w'))
