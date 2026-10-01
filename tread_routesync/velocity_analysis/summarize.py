"""Image-level summaries and paired bootstrap CIs; no optional plotting packages."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .run import write_csv


def bootstrap(values, rng, repeats=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    means = values[rng.integers(0,len(values),(repeats,len(values)))].mean(1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def main(root):
    root=Path(root)
    with (root/'per_image_pairs.csv').open() as f:
        rows=list(csv.DictReader(f))
    metrics=('cosine','l1','rmse','squared_l2_mean')
    image_groups=defaultdict(list)
    pair_counts=defaultdict(int)
    for row in rows:
        key=(row['source'],row['field'],row['scope'],float(row['t']),row['pair_type'])
        image_groups[key+(int(row['sample_id']),)].append([float(row[m]) for m in metrics])
        pair_counts[key]+=int(row['pair_count'])
    group_values=defaultdict(dict)
    for key, values in image_groups.items():
        group_values[key[:-1]][key[-1]]=np.mean(values,axis=0)
    rng=np.random.default_rng(20260922)
    summaries=[]
    for key, images in sorted(group_values.items()):
        values=np.array(list(images.values()))
        for j,metric in enumerate(metrics):
            valid=values[:,j][np.isfinite(values[:,j])]
            if not len(valid): continue
            ci=bootstrap(valid,rng)
            summaries.append(dict(source=key[0],field=key[1],scope=key[2],t=key[3],pair_type=key[4],metric=metric,
                mean=float(valid.mean()),ci_low=ci[0],ci_high=ci[1],images=len(valid),total_pairs=pair_counts[key]))
    write_csv(root/'summary.csv',summaries)
    contrasts=[]
    for key in sorted({k[:-1] for k in group_values}):
        rp=group_values[key+('R-P',)]; rr=group_values[key+('R-R',)]; pp=group_values[key+('P-P',)]
        ids=sorted(set(rp)&set(rr)&set(pp))
        rv=np.array([rp[i] for i in ids]); within=(np.array([rr[i] for i in ids])+np.array([pp[i] for i in ids]))/2
        for j,metric in enumerate(metrics):
            d=rv[:,j]-within[:,j]; valid=np.isfinite(d); d=d[valid]
            if not len(d): continue
            ci=bootstrap(d,rng)
            baseline=float(within[valid,j].mean())
            contrasts.append(dict(source=key[0],field=key[1],scope=key[2],t=key[3],metric=metric,
                rp_mean=float(rv[valid,j].mean()),within_mean=baseline,difference=float(d.mean()),
                relative_percent=float(100*d.mean()/baseline) if abs(baseline)>1e-12 else None,
                ci_low=ci[0],ci_high=ci[1],images=len(d)))
    write_csv(root/'paired_contrasts.csv',contrasts)
    with (root/'per_image_fm.csv').open() as f: fm=list(csv.DictReader(f))
    by_image=defaultdict(list)
    for row in fm: by_image[(float(row['t']),row['metric'],row['group'],int(row['sample_id']))].append(float(row['value']))
    stats=defaultdict(dict)
    for key,value in by_image.items(): stats[key[:-1]][key[-1]]=np.mean(value)
    fm_summary=[]
    for t,metric in sorted({key[:2] for key in stats}):
        r=stats[(t,metric,'R')]; p=stats[(t,metric,'P')]; ids=sorted(set(r)&set(p))
        a=np.array([r[i] for i in ids]); b=np.array([p[i] for i in ids]); d=a-b; ci=bootstrap(d,rng)
        fm_summary.append(dict(t=t,metric=metric,r_mean=float(a.mean()),p_mean=float(b.mean()),difference=float(d.mean()),
            relative_percent=float(100*d.mean()/b.mean()),ci_low=ci[0],ci_high=ci[1],images=len(d)))
    write_csv(root/'fm_summary.csv',fm_summary)
    payload=dict(summary=summaries,contrasts=contrasts,fm=fm_summary)
    (root/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    html='''<!doctype html><meta charset="utf-8"><title>RouteSync velocity analysis</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1100px;color:#172b4d}select{padding:8px;margin:5px}svg{width:100%;background:#fafbfd;border:1px solid #dde3ee}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}small{color:#526071}</style>
<h1>RouteSync sparse velocity analysis</h1><p>SiT-B/2 · EMA 400K · active 50% · CFG 1 · BF16. R=bypass, P=processed.</p>
<p>Data: 256 training images × 2 routing masks × 7 noise levels. Rollout: 32 Euler trajectories, 50 steps, fixed routing per trajectory. t=0 clean; t=1 noise.</p>
<div id="controls"></div><svg id="chart" viewBox="0 0 1000 440"></svg><p id="caption"></p><table id="table"></table>
<small>95% paired image-bootstrap confidence intervals; masks averaged within image. These are exploratory, pointwise intervals, not corrected for multiple comparisons. Velocity difference alone is not prediction error. All-pair and neighboring-patch vectors have 16 components; boundary pixels have 4. Do not compare cosine levels directly across these scopes.</small>
<script>const data=PAYLOAD;
const specs={source:['noised_data','generated_trajectory'],field:['prediction','target','error'],scope:['all_pairs','adjacent_patches','boundary_pixels','distance_1_to_2','distance_2_to_6','distance_gt_6'],metric:['cosine','l1','rmse','squared_l2_mean']};
for(const [name,values] of Object.entries(specs)){const label=document.createElement('label');label.textContent=name+' ';const s=document.createElement('select');s.id=name;values.forEach(v=>s.add(new Option(v,v)));s.onchange=draw;label.append(s);document.querySelector('#controls').append(label)}
function draw(){const q=Object.fromEntries(Object.keys(specs).map(k=>[k,document.getElementById(k).value]));const rows=data.summary.filter(r=>Object.keys(q).every(k=>r[k]===q[k]));const svg=document.getElementById('chart');if(!rows.length){svg.innerHTML='<text x="80" y="90">No target/error available on generated trajectories.</text>';document.getElementById('table').innerHTML='';return}
const min=Math.min(...rows.map(r=>r.ci_low)),max=Math.max(...rows.map(r=>r.ci_high)),pad=(max-min)*.15||.01,lo=min-pad,hi=max+pad;const X=t=>80+t*850,Y=v=>365-(v-lo)/(hi-lo)*310;let s='';for(let i=0;i<=5;i++){const val=lo+(hi-lo)*i/5,yy=Y(val);s+=`<path d="M80 ${yy}H930" stroke="#e0e5ef"/><text x="72" y="${yy+5}" text-anchor="end" font-size="12">${val.toFixed(4)}</text>`;const xx=X(i/5);s+=`<text x="${xx}" y="393" text-anchor="middle" font-size="13">${(i/5).toFixed(1)}</text>`}s+='<text x="470" y="424">t (0=clean, 1=noise)</text>';
['R-R','R-P','P-P'].forEach((g,k)=>{const color=['#2563eb','#dc2626','#059669'][k],rr=rows.filter(r=>r.pair_type===g).sort((a,b)=>a.t-b.t);s+=`<text x="${110+k*150}" y="27" fill="${color}" font-weight="bold">${g}</text><polyline fill="none" stroke="${color}" stroke-width="2.5" points="${rr.map(r=>X(r.t)+','+Y(r.mean)).join(' ')}"/>`;rr.forEach(r=>s+=`<path d="M${X(r.t)} ${Y(r.ci_low)}V${Y(r.ci_high)}" stroke="${color}"/><circle cx="${X(r.t)}" cy="${Y(r.mean)}" r="4" fill="${color}"><title>${g} t=${r.t} mean=${r.mean.toFixed(6)}</title></circle>`)});svg.innerHTML=s;
const cc=data.contrasts.filter(r=>Object.keys(q).every(k=>r[k]===q[k]));document.getElementById('caption').textContent='Below: R-P minus average(R-R, P-P), paired within each image.';document.getElementById('table').innerHTML='<tr><th>t</th><th>R-P</th><th>Within-group mean</th><th>Difference</th><th>95% CI</th><th>Relative %</th></tr>'+cc.map(r=>`<tr><td>${r.t}</td><td>${r.rp_mean.toFixed(5)}</td><td>${r.within_mean.toFixed(5)}</td><td>${r.difference.toFixed(5)}</td><td>[${r.ci_low.toFixed(5)}, ${r.ci_high.toFixed(5)}]</td><td>${r.relative_percent?.toFixed(2)}</td></tr>`).join('')};draw();</script>'''
    (root/'dashboard.html').write_text(html.replace('PAYLOAD',json.dumps(payload,allow_nan=False)))
    print('Wrote summary.csv, paired_contrasts.csv, fm_summary.csv, summary.json, dashboard.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');main(p.parse_args().output)
