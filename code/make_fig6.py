import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
r=json.load(open('robustness.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
s=r['shuffle']
null=np.random.RandomState(0)  # placeholder not used
ax[0].axvspan(s['null_mean']-2*s['null_sd'],s['null_mean']+2*s['null_sd'],color='#BDD7EE',label='Shuffled labels (mean ± 2 SD, 20 permutations)')
ax[0].axvline(s['null_mean'],color='#2E75B6',lw=1)
ax[0].axvline(s['null_max'],color='#2E75B6',lw=1,ls=':',label=f"Best shuffled run (R² = {s['null_max']:.2f})")
ax[0].axvline(s['real_mean'],color='#C55A11',lw=2,label=f"Real labels (R² = {s['real_mean']:.2f})")
ax[0].set_xlim(-0.35,0.25); ax[0].set_yticks([]); ax[0].set_xlabel('Random-split CV R² (Random Forest)')
ax[0].legend(frameon=False,fontsize=6.8,loc='upper left'); ax[0].set_title('a  Label-shuffling control',loc='left',fontsize=9,fontweight='bold')
lc=r['learning_curve']; fr=sorted(lc,key=float); x=[float(f)*100 for f in fr]
for key,col,lab in [('random','#C55A11','Random split'),('grouped','#2E75B6','Studies held out')]:
    m=[lc[f][f'{key}_R2'] for f in fr]; sd=[lc[f][f'{key}_sd']/np.sqrt(20) for f in fr]
    ax[1].errorbar(x,m,yerr=sd,marker='o',ms=4,color=col,capsize=2,label=lab,lw=1.3)
ax[1].axhline(0,color='k',lw=0.6); ax[1].set_xticks(x); ax[1].set_xlabel('Training data used (% of training studies)'); ax[1].set_ylabel('Test R² (Random Forest)')
ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('b  Learning curve',loc='left',fontsize=9,fontweight='bold')
plt.tight_layout(); plt.savefig('fig6.png',dpi=300)
