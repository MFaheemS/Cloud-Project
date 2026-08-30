"""Fit heat-up / cool-down time constants and derive the fleet-size bound."""
import csv, sys, statistics as st

def load(tag, d="experiments/raw"):
    tel=[(float(r[0]),r[1],float(r[2]),float(r[3]),float(r[4]),int(r[5]))
         for r in list(csv.reader(open(f"{d}/tc_{tag}_tel.csv")))[1:]]
    reqs=[[float(x) for x in r] for r in list(csv.reader(open(f"{d}/tc_{tag}_reqs.csv")))[1:]]
    return tel,reqs

def tau(series,t0,start,steady):
    """time to reach 63.2% of the transition start->steady"""
    tgt=start+0.632*(steady-start)
    for t,v in series:
        if (steady>start and v>=tgt) or (steady<start and v<=tgt): return t-t0
    return None

def analyze(tag):
    tel,reqs=load(tag)
    ld=[r for r in tel if r[1]=="load"]; cl=[r for r in tel if r[1]=="cool"]
    l0,c0=ld[0][0],cl[0][0]
    tl=[(r[0],r[2]) for r in ld]; tc=[(r[0],r[2]) for r in cl]
    ts,te=st.mean([v for _,v in tl[:40]]), st.mean([v for _,v in tl[-200:]])
    cs,ce=st.mean([v for _,v in tc[:20]]), st.mean([v for _,v in tc[-200:]])
    th=tau(tl,l0,ts,te); tcool=tau(tc,c0,cs,ce)
    tk=[(r[0],r[3]) for r in reqs]
    ks,ke=st.mean([r[3] for r in reqs[:8]]), st.mean([r[3] for r in reqs[-15:]])
    tk_tau=tau(tk,0,ks,ke)
    return dict(tag=tag,tau_heat=th,tau_cool=tcool,ratio=tcool/th if th else None,
                tau_tput=tk_tau,tput_start=ks,tput_end=ke,
                loss_pct=(1-ke/ks)*100,temp_start=ts,temp_end=te,
                blind_ratio=tk_tau/th if th and tk_tau else None,
                n_min=1+tcool/th if th else None)

if __name__=="__main__":
    for tag in (sys.argv[1:] or ["q4"]):
        d=analyze(tag)
        print(f"\n=== {tag} ===")
        print(f"  throughput   {d['tput_start']:.1f} -> {d['tput_end']:.1f} tok/s ({d['loss_pct']:.1f}% loss)")
        print(f"  temp         {d['temp_start']:.1f} -> {d['temp_end']:.1f} C")
        print(f"  tau_heat(T)  {d['tau_heat']:.0f}s   tau_cool(T) {d['tau_cool']:.0f}s   ratio {d['ratio']:.2f}")
        print(f"  tau_heat(tput) {d['tau_tput']:.0f}s  -> temp saturates {d['blind_ratio']:.1f}x faster than tput")
        print(f"  min fleet size N >= {d['n_min']:.1f}")
