import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
r=pd.read_csv('results_descriptor_run/yield_regression_results.csv'); s=pd.read_csv('results_descriptor_run/selectivity_classification_results.csv')
sets=['A: conditions','B: + RDKit descriptors','C: + Morgan fingerprints']; lab=['A\nconditions','B\n+ descriptors','C\n+ fingerprints']
cols={'Ridge':'#2E75B6','Logistic regression':'#2E75B6','Random Forest':'#C55A11','XGBoost':'#548235'}
fig,ax=plt.subplots(1,2,figsize=(7.4,3.3))
def panel(a,df,models,schemes,metric,sd,ylab,title,ref):
    w=0.13; xs=np.arange(len(sets))
    k=0
    for m in models:
        for sc in schemes:
            sub=df[(df.model==m)&(df.validation==sc)].set_index('feature_set').loc[sets]
            off=(k-(len(models)*2-1)/2)*w
            err=sub[sd].values if sd in sub and sub[sd].notna().all() else None
            a.bar(xs+off,sub[metric].values,w*0.92,yerr=err,capsize=1.5,color=cols[m],
                  alpha=1 if sc=='random' else 0.45,edgecolor=cols[m],hatch=None if sc=='random' else '////',lw=0.6,error_kw=dict(lw=0.7))
            k+=1
    a.axhline(ref,color='k',lw=0.7,ls='--' if ref else '-')
    a.set_xticks(xs); a.set_xticklabels(lab,fontsize=7.5); a.set_ylabel(ylab); a.set_title(title,loc='left',fontsize=9,fontweight='bold')
panel(ax[0],r,['Ridge','Random Forest','XGBoost'],['random','grouped'],'R2_mean','R2_sd','Cross-validated R²','a  Yield (n = 239)',0)
panel(ax[1],s,['Logistic regression','Random Forest','XGBoost'],['random','LOSO'],'AUC_mean','AUC_sd','ROC-AUC','b  Selectivity ≥ 0.90 (n = 76)',0.5)
ax[1].set_ylim(0.3,0.8)
h=[Patch(color=cols['Ridge'],label='Ridge / logistic'),Patch(color=cols['Random Forest'],label='Random Forest'),Patch(color=cols['XGBoost'],label='XGBoost'),
   Patch(facecolor='white',edgecolor='grey',label='Random CV (solid)'),Patch(facecolor='lightgrey',edgecolor='grey',hatch='////',label='Study held out (hatched)')]
fig.legend(handles=h,loc='lower center',ncol=5,frameon=False,fontsize=7.2)
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig('fig4.png',dpi=300)
