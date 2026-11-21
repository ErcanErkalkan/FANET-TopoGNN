"""Rebuild five provenance-aware, vector journal figures from the frozen summary tables.

This is a presentation-only step. It does not update frozen simulations or scores.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import patheffects as pe

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_derived'
OUT=ROOT/'paper/manuscript/figures'
OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})

def rows(filename):
    with (DATA/filename).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def export(fig,slug):
    fig.savefig(OUT/(slug+'.pdf'),bbox_inches='tight',pad_inches=.06,
                metadata={'Creator':'FANET-TopoGNN reproducible figure renderer','CreationDate':None,'ModDate':None})
    fig.savefig(OUT/(slug+'.png'),dpi=200,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

# Figure 1: method schematic faithfully mirrors code components and the fixed protocol.
fig,ax=plt.subplots(figsize=(8.2,3.35));ax.set_xlim(0,12.4);ax.set_ylim(0,5);ax.axis('off')
def block(x,y,w,h,lines,shade='#f1f3f5',fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',lw=0.9,
                                ec='#404040',fc=shade))
    ax.text(x+w/2,y+h/2,lines,ha='center',va='center',fontsize=fs,linespacing=1.2)
def arrow(start,end,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,linewidth=1.15,
                                 linestyle='--' if dashed else '-',color='#303030'))
block(.1,1.55,1.73,1.3,'Current 2-D state\npositions, velocity\nlink graph')
block(2.35,2.95,1.65,1.05,'Two-layer GCN\n64-d graph state')
block(2.35,.55,1.65,1.05,'16×16 persistence\nimage encoder')
block(4.55,1.55,1.54,1.3,'Gated fusion\n64-d representation')
block(6.62,1.55,1.56,1.3,'Regression\nβ₀(t+6)')
block(8.72,1.55,1.53,1.3,'Logistic\nendpoint score')
block(10.7,1.55,1.55,1.3,'Fixed threshold\nτ = 0.15')
block(10.7,.12,1.55,.72,'Alert logic\n& cooldown',shade='#e2e9ef')
arrow((1.83,2.2),(2.35,3.46));arrow((1.83,2.2),(2.35,1.08));arrow((4,3.48),(4.55,2.45));arrow((4,1.08),(4.55,1.88))
arrow((6.09,2.2),(6.62,2.2));arrow((8.18,2.2),(8.72,2.2));arrow((10.25,2.2),(10.7,2.2));arrow((11.48,1.55),(11.48,.85))
ax.text(1.98,4.64,'Single decision snapshot (not a temporal GNN)',ha='left',fontsize=9,fontweight='bold')
ax.text(6.6,.78,'Calibration label: graph fragmented at t+6',fontsize=8)
ax.text(6.6,.33,'Evaluation target: onset within six frames',fontsize=8,fontstyle='italic')
export(fig,'fig1_warning_pipeline')

# Figure 2: explicitly conceptual event definition, no simulated trajectory implied.
fig,ax=plt.subplots(figsize=(8.2,2.55));ax.set_xlim(-.5,6.65);ax.set_ylim(-.25,2.6)
ax.set_xticks(range(7));ax.set_xticklabels(['t','t+1','t+2','t+3','t+4','t+5','t+6']);ax.set_yticks([])
ax.set_xlabel('Illustrative sampled frame (Δt = 0.1 s)')
states=[1,1,2,2,1,1,1]
for k,state in enumerate(states):
    x=k-.37; fc='#d7d7d7' if state>1 else '#ffffff'
    ax.add_patch(Rectangle((x,1.25),.74,.55,facecolor=fc,edgecolor='#353535',lw=1))
    ax.text(k,1.52,'split' if state>1 else 'linked',ha='center',va='center',fontsize=8)
ax.annotate('Onset at t+2',xy=(2,1.82),xytext=(1.45,2.35),arrowprops=dict(arrowstyle='->'),fontsize=8)
ax.annotate('Reconnected by t+4',xy=(4,1.82),xytext=(3.75,2.35),arrowprops=dict(arrowstyle='->'),fontsize=8)
ax.text(0,.72,'Endpoint label at t+6 = 0 (connected)',fontsize=9)
ax.text(0,.27,'Window-onset label = 1 (fragmentation starts at t+2)',fontsize=9)
ax.text(3.15,-.12,'Schematic only: not a recorded simulation run',ha='center',fontsize=8,fontstyle='italic')
for spine in ax.spines.values():spine.set_visible(False)
ax.tick_params(axis='x',length=0)
export(fig,'fig2_target_mismatch_schematic')

# Figure 3: mobility x node count, two complementary measures; all cells have equal planned exposure.
r=rows('wn_r9_10_by_mobility_node_count.csv');mob=['rwp','gm','mission'];n=[10,20,30]
lookup={(v['mobility'],int(v['n_nodes'])):v for v in r}
assert len(lookup)==9
fpr=np.array([[float(lookup[(m,k)]['false_alerts_per_minute']) for k in n] for m in mob])
rec=np.array([[100*float(lookup[(m,k)]['event_recall']) for k in n] for m in mob])
assert np.isclose(fpr.sum(),(2.55+0.15+5.85+1.50+3.15),atol=.00001)
fig,axes=plt.subplots(1,2,figsize=(8.1,3.2),gridspec_kw={'wspace':.30})
for ax,mat,cap,fmt,vmax in [(axes[0],fpr,'(a) False alarms per full simulated minute','{:.2f}',6.0),
                              (axes[1],rec,'(b) Event recall (%)','{:.1f}',50.0)]:
    ax.set_xlim(-.5,2.5);ax.set_ylim(2.5,-.5)
    for ri in range(3):
      for cj in range(3):
        fill=plt.get_cmap('Greys')(float(mat[ri,cj])/vmax)
        ax.add_patch(Rectangle((cj-.5,ri-.5),1,1,facecolor=fill,edgecolor='#999999',lw=.6))
    ax.set_xticks(range(3));ax.set_xticklabels([str(k) for k in n]);ax.set_xlabel('Number of nodes')
    ax.set_yticks(range(3));ax.set_yticklabels(['RWP','Gauss–Markov','Mission']);ax.set_title(cap,fontsize=10)
    for i in range(3):
      for j in range(3):
        val=mat[i,j]; s=fmt.format(val); color='white' if val>(.43*vmax) else '#171717'
        if ax is axes[0] and val>2: s+='*'
        ax.text(j,i,s,ha='center',va='center',color=color,fontsize=10,fontweight='bold' if (ax is axes[0] and val>2) else 'normal')
    ax.set_xticks(np.arange(-.5,3,.999999),minor=True);ax.set_yticks(np.arange(-.5,3,.999999),minor=True)
    ax.grid(which='minor',color='#999999',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
fig.text(.03,-.015,'* > 2 false alarms/min reference; each cell: 5 trajectory groups, 20 runs and 6.67 simulated min.',fontsize=8)
export(fig,'fig3_mobility_node_heatmaps')

# Figure 4: 15 cell grid and very prominent over-budget markings.
r=rows('wn_r9_10_all_strata.csv');b=[2,3,4,5,6]; q={(int(x['replicate']),x['mobility']):x for x in r if x['scope']=='replicate_mobility'}
assert len(q)==15
rate=np.array([[float(q[(k,m)]['false_alerts_per_minute']) for m in mob] for k in b])
assert sum(rate.flatten()>2)==6
fig,ax=plt.subplots(figsize=(7.2,3.55));ax.set_xlim(-.5,2.5);ax.set_ylim(4.5,-.5)
for ri in range(5):
 for cj in range(3):
  fill=plt.get_cmap('Greys')(rate[ri,cj]/4)
  ax.add_patch(Rectangle((cj-.5,ri-.5),1,1,facecolor=fill,edgecolor='#a0a0a0',lw=.6))
ax.set_xticks(range(3));ax.set_xticklabels(['Random waypoint','Gauss–Markov','Mission-based']);ax.set_yticks(range(5));ax.set_yticklabels([f'Block {k}' for k in b])
ax.set_xlabel('Mobility');ax.set_ylabel('Development evaluation block')
for i in range(5):
 for j in range(3):
    x=rate[i,j];ok=x>2
    ax.text(j,i,f'{x:.2f}'+('*' if ok else ''),ha='center',va='center',fontsize=11,
            color='white' if x>1.8 else '#111111',weight='bold' if ok else 'normal')
    if ok: ax.add_patch(Rectangle((j-.47,i-.46),.94,.92,fill=False,edgecolor='black',lw=1.6,linestyle='--'))
ax.set_xticks(np.arange(-.5,3,.999999),minor=True);ax.set_yticks(np.arange(-.5,5,.999999),minor=True)
ax.grid(which='minor',color='#a0a0a0',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
fig.text(.11,.005,'* Budget exceedance (> 2/min); each cell: 3 trajectory groups, 12 paired runs, 4 simulated min.',fontsize=8)
export(fig,'fig4_block_mobility_budget')

# Figure 5: alert lead and unverified hypothetical action latency are visually separated.
leads=[v for v in rows('wn_r9_10_lead_distribution.csv') if v['method']=='TopoGNN']; leads.sort(key=lambda x:int(x['lead_ticks']))
t=[float(v['lead_seconds']) for v in leads];counts=[int(v['matched_alerts']) for v in leads]
a=[v for v in rows('wn_r9_10_action_latency_sensitivity.csv') if v['method']=='TopoGNN' and float(v['assumed_effective_action_latency_seconds'])>0];a.sort(key=lambda x:float(x['assumed_effective_action_latency_seconds']))
assert sum(counts)==183 and [int(v['potentially_timely_matched_alerts']) for v in a]==[183,98,58,33,20,7]
fig,axes=plt.subplots(1,2,figsize=(8.1,3.0),gridspec_kw={'wspace':.30})
ax=axes[0];bars=ax.bar(t,counts,width=.071,edgecolor='#222222',linewidth=.8,facecolor='#bcbcbc');ax.set_xticks(t);ax.set_xticklabels([f'{v:.1f}' for v in t]);ax.set_xlabel('Lead time (s)');ax.set_ylabel('Matched warnings');ax.set_ylim(0,99);ax.set_title('(a) Recorded lead time',fontsize=10)
for bar,n0 in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(n0),ha='center',fontsize=8)
ax=axes[1];x=[float(v['assumed_effective_action_latency_seconds']) for v in a];y=[int(v['potentially_timely_matched_alerts']) for v in a]
ax.plot(x,y,marker='o',linewidth=1.7,color='#333333',markerfacecolor='white',markersize=6)
ax.set_xticks(x);ax.set_xlabel('Assumed action latency (s)');ax.set_ylabel('Pairs with sufficient lead');ax.set_ylim(0,205);ax.set_title('(b) Hypothetical timing only',fontsize=10)
for xv,yv in zip(x,y):ax.annotate(str(yv),(xv,yv),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
fig.text(.04,.005,'No controller or mitigation was simulated. Matched warnings = 183; total fragmentation onsets = 1,114.',fontsize=8)
export(fig,'fig5_lead_action_latency')

assert len(list(OUT.glob('fig*.pdf')))==5
print('Generated five PDF vector figures from frozen summary tables + two labeled method schematics.')
