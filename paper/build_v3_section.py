"""Fill sec_v3_real.tmpl from results_v3 CSVs and wire it (plus the parity appendix) into main.tex."""
import os, pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D=pd.read_csv('results_v3/real_seg.csv'); R=pd.read_csv('results_v3/real_size.csv'); E=pd.read_csv('results_v3/ant_engine_summary.csv'); T=pd.read_csv('results_v3/ant_tasks.csv'); O=pd.read_csv('results_v3/official_baselines.csv')
EL="\\\\"; esc=lambda s: str(s).replace("_","\\_")
lab={"real_main":"Kaggle, natural mix","real_m4":"Kaggle, $M{=}4$ plan (v1)","real_m4v2":"Kaggle, $M{=}4$ plan (v2)","real_m8":"Kaggle, $M{=}8$ plan (v1)","real_m8v2":"Kaggle, $M{=}8$ plan (v2)","real_m4v3":"Kaggle, $M{=}4$ parallel","real_m8v3":"Kaggle, $M{=}8$ parallel","fw_langgraph":"LangGraph agent","fw_autogen":"AutoGen agent",
     "ftc_gemini":"Gemini, day 1 (clean)","ftc_gemini_d2":"Gemini, day 2 (clean)","ftc_groq":"Groq, day 1 (clean)","ftc_groq_d2":"Groq, day 2 (clean)","ftc_cloudflare":"Workers AI (clean)","ft_gemini":"Gemini (free tier)"}
SEGTAGS=[k for k in lab if not k.startswith("ft_")]
seg=[]
for t in SEGTAGS:
    a=D[(D.tag==t)&(D.clock=="wire")&(D.method=="frame_rule (testbed)")]; b=D[(D.tag==t)&(D.clock=="wire")&(D.method=="burst_rule (real)")]
    if len(a): seg.append("%s & %d & %.3f & %.3f [%.3f, %.3f] %s"%(lab[t],b.n_req.iloc[0],a["F1@100"].iloc[0],b["F1@100"].iloc[0],b["F1@100_lo"].iloc[0],b["F1@100_hi"].iloc[0],EL))
size=[]
for t in ["real_main","real_m4v3","real_m8v3","fw_langgraph","fw_autogen","ft_gemini"]:
    for reg,rl in [("<1KB","$<$1\\,KB"),(">=1KB","$\\geq$1\\,KB")]:
        x=R[(R.tag==t)&(R.estimator=="struct_prop1")&(R.regime==reg)]; c=R[(R.tag==t)&(R.estimator=="lgbm_q50 + CQR")&(R.regime==reg)]
        if len(x): size.append("%s & %s & %d & %.1f & %.1f & %.2f %s"%(lab[t],rl,x.n.iloc[0],100*x.MedAPE.iloc[0],100*c.MedAPE.iloc[0],c.PICP90.iloc[0],EL))
ant=["%s & %d & %d & %.2f & %.0f & %.1f & %.0f %s"%(esc(r.engine),r.episodes,r.requests,r.spearman_nreq_vs_asst,r.req_kB_med,100*r.shrink_frac,100*r.eps_with_shrink,EL) for r in E.itertuples()]
tk=["%s & %s & %.3f & %.3f %s"%(esc(r.target),esc(r.features),r.acc,r.macroF1,EL) for r in T.itertuples()]
ob=["%s & %s & %.3f %s"%(esc(r.model),esc(r.tag),r.macroF1,EL) for r in O.itertuples()]
NM={"real_main":"Kaggle, dedicated","real_m4v3":"Kaggle, $M{=}4$","real_m8v3":"Kaggle, $M{=}8$","fw_langgraph":"LangGraph","fw_autogen":"AutoGen","ft_gemini":"Gemini (per-request conn.)"}
def track_rows():
    Lp=pd.read_csv('results_v3/tracker_link_profiled.csv'); Sp=pd.read_csv('results_v3/tracker_size_profiled.csv'); out=[]
    for c,l in NM.items():
        g=lambda mth: Lp[(Lp.corpus==c)&(Lp.method==mth)].acc.iloc[0]
        z=lambda mth: Sp[(Sp.corpus==c)&(Sp.method==mth)&(Sp.regime==">=1KB")].MedAPE_per_corpus_fit_odd_eps
        zp, zj = z("prev_on_connection"), z("joint_assignment")
        out.append("%s & %.2f & %.2f & %s & %.1f\\%% \\\\"%(l,g("prev_on_connection"),g("joint_assignment"),("%.1f\\%%"%(100*zp.iloc[0])) if len(zp) and zp.notna().iloc[0] else "--",100*zj.iloc[0]))
    return out
def size_rows():
    Sp=pd.read_csv('results_v3/tracker_size_profiled.csv'); Q=pd.read_csv('results_v3/cqr_percorpus.csv'); out=[]
    for c,l in NM.items():
        for reg,rl in (("<1KB","$<$1\\,KB"),(">=1KB","$\\geq$1\\,KB")):
            x=Sp[(Sp.corpus==c)&(Sp.method=="joint_assignment")&(Sp.regime==reg)]; q=Q[(Q.corpus==c)&(Q.scheme=="per_corpus_x_size")&(Q.regime==reg)]
            if len(x) and x.MedAPE_per_corpus_fit_odd_eps.notna().iloc[0]:
                out.append("%s & %s & %d & %.1f & %s \\\\"%(l,rl,x.n_odd.iloc[0],100*x.MedAPE_per_corpus_fit_odd_eps.iloc[0],("%.2f"%q.PICP90.iloc[0]) if len(q) else "--"))
    return out
s=open('paper_tex/sec_v3_real.tmpl').read()
for k,v in zip(["@SEG@","@ANT@","@TK@","@OB@"],(seg,ant,tk,ob)): s=s.replace(k,"\n".join(v))
TRACK=track_rows(); s=s.replace('@SIZE@','\n'.join(size_rows()))
s=s.replace('@TRACK@','\n'.join(TRACK))
open('paper_tex/sec_v3_real.tex','w').write(s)
APP=r"""\appendices
\section{Baseline hyper-parameter parity}\label{app:parity}
\noindent{\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{0.18\columnwidth}p{0.34\columnwidth}p{0.36\columnwidth}}\toprule
Baseline & Source & Settings used\\\midrule
Whisper-Leak & {\footnotesize\texttt{yo-yo-yo-jbo/}\newline\texttt{whisper\_leak}} & LightGBM, 5000 trees, learning rate 0.02, early stopping 40, max length 584 (p95), multiclass objective\\
Deep Fingerprinting & {\footnotesize\texttt{deep-}\newline\texttt{fingerprinting/df}} & DFNet unchanged; 5000 directions; Adamax, learning rate 0.002, batch 128, 30 epochs\\\bottomrule
\end{tabular}}
"""
m=open('paper_tex/main.tex').read()
if 'sec_v3_real' not in m: m=m.replace('\\section{Limitations}','\\input{sec_v3_real}\n\n\\section{Limitations}')
if 'app:parity' not in m: m=m.replace('\\begin{thebibliography}', APP+'\n\\begin{thebibliography}')
m=m.replace('\\usepackage{graphicx,booktabs,multirow}','\\usepackage{graphicx,booktabs,multirow}') 
open('paper_tex/main.tex','w').write(m); print(s.count('@'), len(seg),len(size),len(ant),len(tk),len(ob))
