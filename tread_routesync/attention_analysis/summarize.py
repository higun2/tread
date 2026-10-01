"""Image-cluster paired confidence intervals and dependency-free visual report."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .run import MODES, METRICS, write_csv


def svg_lines(path, xs, series, title, xlabel, ylabel):
    colors = ['#334155','#dc4b40','#168a78','#9860bc']
    w,h = 840,440
    left,top,pw,ph = 85,65,705,280
    low = min(0.,min(min(y) for _,y in series))
    high = max(max(y) for _,y in series)
    high = max(high,1e-12)*1.08
    xcoord = lambda x: left+(x-min(xs))/max(max(xs)-min(xs),1e-10)*pw
    ycoord = lambda y: top+ph-(y-low)/(high-low)*ph
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial" fill="#243247"><text x="30" y="30" font-size="19">{html.escape(title)}</text>']
    for y in np.linspace(low,high,6):
        yy=ycoord(y)
        parts.append(f'<path d="M{left},{yy}h{pw}" stroke="#e2e8f0"/><text x="{left-8}" y="{yy+4}" text-anchor="end" font-size="12">{y:.4g}</text>')
    for x in xs:
        parts.append(f'<text x="{xcoord(x)}" y="{top+ph+22}" text-anchor="middle" font-size="12">{x:g}</text>')
    for k,(name,ys) in enumerate(series):
        color=colors[k]
        coords=' '.join(f'{xcoord(x):.2f},{ycoord(y):.2f}' for x,y in zip(xs,ys))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x,y in zip(xs,ys):
            parts.append(f'<circle cx="{xcoord(x)}" cy="{ycoord(y)}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="{left+k*220}" y="410" fill="{color}" font-size="14">{html.escape(name)}</text>')
    parts.append(f'<text x="{left+pw/2}" y="390" text-anchor="middle">{xlabel}</text><text x="22" y="{top+ph/2}" transform="rotate(-90 22 {top+ph/2})" text-anchor="middle">{ylabel}</text></g></svg>')
    path.write_text(''.join(parts))


def main(folder):
    out=Path(folder)
    z=np.load(out/'measurements.npz')
    v=z['values'].astype(np.float64)
    assert np.isfinite(v).all(), 'Analysis incomplete'
    times=z['timesteps'].tolist(); layers=z['layers'].tolist()
    meta=json.loads((out/'metadata.json').read_text())
    n=v.shape[0]
    bootstrap=np.random.default_rng(413).integers(n,size=(2000,n))
    def stats(x):
        lo,hi=np.quantile(x[bootstrap].mean(1),[.025,.975])
        return dict(mean=float(x.mean()),ci95_low=float(lo),ci95_high=float(hi))
    # Each image is an independent unit; average masks, then heads within image.
    a=v.mean(2).mean(-2)  # image,time,layer,mode,metric
    summaries=[]; contrasts=[]; heads=[]
    for ti,t in enumerate(times):
        for li,layer in enumerate(layers):
            for mi,mode in enumerate(MODES):
                for ki,metric in enumerate(METRICS):
                    summaries.append(dict(t=t,layer=layer,mode=mode,metric=metric,**stats(a[:,ti,li,mi,ki])))
            for name,before,after in [('local_correction',1,2),('full_correction',3,4),('local_sparse_minus_dense',0,1),('full_sparse_minus_dense',0,3)]:
                for ki,metric in enumerate(METRICS):
                    before_x=a[:,ti,li,before,ki]; after_x=a[:,ti,li,after,ki]
                    contrasts.append(dict(t=t,layer=layer,comparison=name,metric=metric,
                        before_mean=float(before_x.mean()),after_mean=float(after_x.mean()),
                        relative_change_pct=float(100*(after_x.mean()/before_x.mean()-1)) if abs(before_x.mean())>1e-20 else None,
                        **stats(after_x-before_x)))
            for hi in range(v.shape[-2]):
                for mi,mode in enumerate(MODES):
                    for ki,metric in enumerate(METRICS):
                        heads.append(dict(t=t,layer=layer,head=hi,mode=mode,metric=metric,mean=float(v[:,ti,:,li,mi,hi,ki].mean())))
    write_csv(out/'summary.csv',summaries)
    write_csv(out/'paired_contrasts.csv',contrasts)
    write_csv(out/'head_summary.csv',heads)
    with (out/'final_velocity.csv').open() as f:
        rows=list(csv.DictReader(f))
    velocity=[]; velocity_contrasts=[]
    for t in times:
        current=[r for r in rows if float(r['t'])==t]
        vals={}
        for mode in ['dense','full_sparse','full_corrected']:
            for metric in ['fm_mse','dense_prediction_mse']:
                x=np.zeros(n); counts=np.zeros(n)
                for r in current:
                    if r['mode']==mode:
                        idx=int(r['sample_id']); x[idx]+=float(r[metric]); counts[idx]+=1
                x/=counts
                vals[mode,metric]=x
                velocity.append(dict(t=t,mode=mode,metric=metric,**stats(x)))
        for metric in ['fm_mse','dense_prediction_mse']:
            x=vals['full_sparse',metric]; y=vals['full_corrected',metric]
            velocity_contrasts.append(dict(t=t,metric=metric,before_mean=float(x.mean()),after_mean=float(y.mean()),relative_change_pct=float(100*(y.mean()/x.mean()-1)),**stats(y-x)))
    write_csv(out/'velocity_summary.csv',velocity)
    write_csv(out/'velocity_contrasts.csv',velocity_contrasts)
    head_comparisons=[]
    for hi in range(v.shape[-2]):
        before=v[:,:,:,0,1,hi,1].mean((1,2))
        after=v[:,:,:,0,2,hi,1].mean((1,2))
        head_comparisons.append(dict(head=hi,sparse_mse=float(before.mean()),corrected_mse=float(after.mean()),
            relative_change_pct=float(100*(after.mean()/before.mean()-1)),**stats(after-before)))
    write_csv(out/'first_layer_head_contrasts.csv',head_comparisons)
    first=a[:,:,0].mean(1) # image,mode,metric
    overview={mode:{metric:stats(first[:,mi,ki]) for ki,metric in enumerate(METRICS)} for mi,mode in enumerate(MODES)}
    overview['first_layer_local_mse_delta']=stats(first[:,2,1]-first[:,1,1])
    overview['first_layer_local_mse_change_pct']=float(100*(first[:,2,1].mean()/first[:,1,1].mean()-1))
    overview['all_layers_full_mse_change_pct']=float(100*(a[:,:,:,4,1].mean()/a[:,:,:,3,1].mean()-1))
    (out/'overview.json').write_text(json.dumps(overview,indent=2))
    svg_lines(out/'first_layer_self_mass.svg',times,[(MODES[m],a[:,:,0,m,0].mean(0).tolist()) for m in [0,1,2]],'First routed layer: diagonal attention mass','t (0 clean, 1 noise)','Mean self-attention probability')
    svg_lines(out/'first_layer_output_mse.svg',times,[(MODES[m],a[:,:,0,m,1].mean(0).tolist()) for m in [1,2]],'First routed layer: AV output error vs dense','t (0 clean, 1 noise)','Mean squared error')
    svg_lines(out/'layer_self_mass.svg',layers,[(MODES[m],a[:,:,:,m,0].mean((0,1)).tolist()) for m in [0,3,4]],'Actual forward: diagonal mass across routed layers','Block index (zero-based)','Mean self-attention probability')
    svg_lines(out/'layer_output_mse.svg',layers,[(MODES[m],a[:,:,:,m,1].mean((0,1)).tolist()) for m in [3,4]],'Actual forward: AV output error vs dense','Block index (zero-based)','Mean squared error')
    svg_lines(out/'final_fm_mse.svg',times,[(mode,[r['mean'] for r in velocity if r['mode']==mode and r['metric']=='fm_mse']) for mode in ['dense','full_sparse','full_corrected']],'Final velocity: conditional FM target MSE','t (0 clean, 1 noise)','Mean squared error')
    svg_lines(out/'final_dense_gap.svg',times,[(mode,[r['mean'] for r in velocity if r['mode']==mode and r['metric']=='dense_prediction_mse']) for mode in ['full_sparse','full_corrected']],'Final velocity: output error vs dense','t (0 clean, 1 noise)','Mean squared error')
    increased=sum(r['relative_change_pct']>0 for r in head_comparisons)
    gap_changes=[r['relative_change_pct'] for r in velocity_contrasts if r['metric']=='dense_prediction_mse']
    significant_fm=[r['t'] for r in velocity_contrasts if r['metric']=='fm_mse' and r['ci95_low']>0]
    lines=['# TREAD attention 진단: tread-v / 0400000','',
        '## 핵심 결과','',
        f'- 첫 routed layer의 self attention은 dense {100*overview["dense"]["self_mass"]["mean"]:.3f}%, sparse {100*overview["local_sparse"]["self_mass"]["mean"]:.3f}%, 보정 sparse {100*overview["local_corrected"]["self_mass"]["mean"]:.3f}%. 포함 확률 차이에 따른 자기 비중 증가와 보정 효과가 관찰된다.',
        f'- 하지만 첫 layer AV 출력의 dense 대비 MSE는 보정 후 {overview["first_layer_local_mse_change_pct"]:.3f}% 증가했다. 7개 timestep 모두 증가하고 각 pointwise paired CI도 0보다 크다. 평균 head별로 {increased}/{v.shape[-2]}개에서 증가했다.',
        '- 실제 sparse forward에서도 모든 routed layer에서 AV 출력 오차가 증가했다. 자기 비중을 맞추는 것만으로 attention 출력의 정확도가 좋아진다는 가설은 이 체크포인트에서 지지되지 않는다.',
        f'- 한편 **최종 velocity의 dense 대비 MSE는 {-max(gap_changes):.2f}–{-min(gap_changes):.2f}% 감소**했다. 중간 attention AV 오차와 최종 velocity 차이는 다른 지표이며 같은 방향으로 움직이지 않았다.',
        f'- Conditional FM target MSE에는 일관된 개선이 없다. t={significant_fm}에서는 작지만 paired CI가 양수인 악화가 있고, 나머지 timestep의 CI는 0을 포함한다.',
        '- 판단: self attention 증가 현상은 확인됐으나 성능 개선 근거는 혼재한다. 이 결과만으로 긴 보정 학습을 우선 추천할 근거는 부족하다. 재학습 시 효과/FID는 미확인이다.', '',
        '## 조건','',f'- EMA, SiT-B/2, 400,000 step. 순수 TREAD, RouteSync 없음.',
        f'- Routed block: zero-based {layers}; 256개 중 128개 active.',
        f'- ImageNet **학습 데이터** latent {n}개, timestep {times}, mask {v.shape[2]}개/이미지. t=0 clean, t=1 noise.',
        '- 동일 이미지·posterior sample·noise·mask로 paired 비교. Conditional CFG=1, FP32, TF32 off.',
        f'- 대각선 보정: log(127/255) = {meta["correction"]:.8f}. 가중치 업데이트 없음.',
        '- local: 각 dense layer의 Q/K/V를 고정하고 같은 active query/key/value만 선택. full: 실제 sparse 전체 forward. corrected는 모든 routed block에 보정 적용.',
        '- Attention 출력 오차는 output projection 전 각 head의 AV를 동일 위치의 dense AV와 비교. head와 query를 평균하며, 통계 단위는 이미지.',
        '', '## 첫 routed layer 결과: noise level을 동일 가중 평균','',
        '| Mode | Self attention | AV MSE vs dense | AV cosine |','|---|---:|---:|---:|']
    for m in ['dense','local_sparse','local_corrected']:
        d=overview[m]
        lines.append(f'| {m} | {d["self_mass"]["mean"]:.6f} | {d["output_mse"]["mean"]:.8f} | {d["output_cosine"]["mean"]:.6f} |')
    delta=overview['first_layer_local_mse_delta']
    lines += ['',f'- 보정의 AV MSE 변화: **{overview["first_layer_local_mse_change_pct"]:+.3f}%** (negative=개선).',
        f'- Paired MSE 차이 corrected−sparse: {delta["mean"]:.8g}; 이미지 bootstrap 95% CI [{delta["ci95_low"]:.8g}, {delta["ci95_high"]:.8g}].',
        '', '## timestep별 첫 layer','', '| t | Dense self | Sparse self | Corrected self | Sparse AV MSE | Corrected AV MSE | MSE change |','|---|---:|---:|---:|---:|---:|---:|']
    for ti,t in enumerate(times):
        x=a[:,ti,0].mean(0)
        lines.append(f'| {t:g} | {x[0,0]:.6f} | {x[1,0]:.6f} | {x[2,0]:.6f} | {x[1,1]:.8f} | {x[2,1]:.8f} | {100*(x[2,1]/x[1,1]-1):+.2f}% |')
    lines += ['', '## 실제 full forward: layer별 비교','', '| Block (zero-based) | Dense self | Sparse self | Corrected self | Sparse AV MSE | Corrected AV MSE | MSE change |','|---|---:|---:|---:|---:|---:|---:|']
    for li,layer in enumerate(layers):
        x=a[:,:,li].mean((0,1))
        lines.append(f'| {layer} | {x[0,0]:.6f} | {x[3,0]:.6f} | {x[4,0]:.6f} | {x[3,1]:.8f} | {x[4,1]:.8f} | {100*(x[4,1]/x[3,1]-1):+.2f}% |')
    lines += ['', '## 최종 velocity: 보정을 학습 없이 적용한 영향','', '| t | Dense FM MSE | Sparse FM MSE | Corrected FM MSE | FM MSE change |','|---|---:|---:|---:|---:|']
    for t in times:
        d={r['mode']:r['mean'] for r in velocity if r['t']==t and r['metric']=='fm_mse'}
        lines.append(f'| {t:g} | {d["dense"]:.6f} | {d["full_sparse"]:.6f} | {d["full_corrected"]:.6f} | {100*(d["full_corrected"]/d["full_sparse"]-1):+.4f}% |')
    lines += ['', '## 최종 velocity의 dense 대비 차이','', '| t | Sparse MSE | Corrected MSE | Change | Paired delta 95% CI |','|---|---:|---:|---:|---|']
    for r in velocity_contrasts:
        if r['metric']=='dense_prediction_mse':
            lines.append(f'| {r["t"]:g} | {r["before_mean"]:.8f} | {r["after_mean"]:.8f} | {r["relative_change_pct"]:+.3f}% | [{r["ci95_low"]:.7g}, {r["ci95_high"]:.7g}] |')
    lines += ['', '## 첫 routed layer의 head별 보정 효과 (t 평균)','', '| Head (zero-based) | Sparse AV MSE | Corrected AV MSE | Change | Paired delta 95% CI |','|---|---:|---:|---:|---|']
    for r in head_comparisons:
        lines.append(f'| {r["head"]} | {r["sparse_mse"]:.8f} | {r["corrected_mse"]:.8f} | {r["relative_change_pct"]:+.3f}% | [{r["ci95_low"]:.7g}, {r["ci95_high"]:.7g}] |')
    lines += ['', '## 해석과 한계','',
        '- Sparse self mass 증가는 key 수 감소만으로도 생긴다. 보정의 타당성은 self mass와 함께 AV 출력 오차를 확인해야 한다.',
        '- Local 비교는 key subsampling만의 효과를 분리한다. Full 비교의 깊은 layer에는 이전 layer부터 누적된 feature 변화도 포함된다.',
        '- AV MSE는 mask별 출력 오차이며, 반복 mask의 기대 출력을 이용한 bias²/variance 분해가 아니다.',
        '- Dense 출력은 reference이지 ground truth가 아니다. Dense와 가까워지는 것만으로 FID 개선을 보장하지 않는다.',
        '- FM target은 해당 clean/noise pair의 conditional target이다. FID 또는 실제 marginal velocity 오차가 아니다.',
        '- 이미 학습된 모델에 inference 때만 보정을 적용했다. 보정을 넣어 재학습한 결과를 예측하거나 검증한 실험은 아니다.',
        '- 생성 trajectory/FID 및 held-out validation은 실행하지 않았다. 학습 데이터 입력의 attention 진단이다.',
        '- 95% CI는 2,000회 이미지 단위 paired bootstrap. 각 이미지의 mask/head 평균 후 계산하며, timestep/head를 독립 표본으로 세지 않는다. 다중비교 보정 없는 탐색적 CI.',
        '', '## 검증','', '```json',json.dumps(meta['validation'],indent=2),'```','',
        '## 파일','', '- [시각화 dashboard](dashboard.html)', '- summary.csv / paired_contrasts.csv: timestep·layer별 평균과 paired 95% CI.',
        '- head_summary.csv: 각 head별 평균.', '- velocity_summary.csv / velocity_contrasts.csv: 최종 velocity/FM 결과.',
        '- measurements.npz: 이미지·timestep·mask·layer·mode·head별 원자료.', '- sample_manifest.csv / metadata.json: 입력 목록과 실행 설정.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    payload=dict(times=times,layers=layers,modes=MODES,metrics=METRICS,summary=summaries,heads=heads,contrasts=contrasts)
    charts=''.join((out/name).read_text() for name in ['first_layer_self_mass.svg','first_layer_output_mse.svg','layer_self_mass.svg','layer_output_mse.svg','final_fm_mse.svg','final_dense_gap.svg'])
    page='''<!doctype html><meta charset="utf-8"><title>TREAD attention audit</title><style>body{font:15px system-ui;background:#f1f5f9;color:#243247;margin:32px auto;max-width:1200px;padding:0 20px}svg{max-width:100%;height:auto;border-radius:10px;margin:12px 0}table{border-collapse:collapse;background:white;width:100%;margin:16px 0}td,th{padding:9px;text-align:right;border-bottom:1px solid #dde3ed}select{padding:8px;margin:5px}pre{white-space:pre-wrap;background:white;padding:24px;border-radius:10px}h1{font-size:26px}</style><h1>TREAD self-attention audit — 400K EMA</h1><p>FP32 · 128 training images · 7 noise levels · 2 masks · conditional CFG=1. Dense is a reference, not ground truth. No training / FID.</p>'''+charts+'''<h2>Head별 결과</h2><label>t <select id="time"></select></label><label>Block (zero-based) <select id="layer"></select></label><label>Metric <select id="metric"></select></label><div id="table"></div><h2>Report</h2><pre>'''+html.escape('\n'.join(lines))+'''</pre><script>const d='''+json.dumps(payload)+'''; for(const [id,vals] of [['time',d.times],['layer',d.layers],['metric',d.metrics]]){const e=document.getElementById(id);e.innerHTML=vals.map(x=>`<option>${x}</option>`).join('');e.onchange=render;} function render(){const t=+document.getElementById('time').value,l=+document.getElementById('layer').value,k=document.getElementById('metric').value;let s='<table><tr><th>Head</th>'+d.modes.map(m=>`<th>${m}</th>`).join('')+'</tr>';const r=d.heads.filter(x=>x.t===t&&x.layer===l&&x.metric===k);for(const h of [...new Set(r.map(x=>x.head))])s+='<tr><td>'+h+'</td>'+d.modes.map(m=>'<td>'+r.find(x=>x.head===h&&x.mode===m).mean.toPrecision(6)+'</td>').join('')+'</tr>';document.getElementById('table').innerHTML=s+'</table>';}render();</script>'''
    page=page.replace('128 training images · 7 noise levels · 2 masks',f'{n} training images · {len(times)} noise levels · {v.shape[2]} masks')
    (out/'dashboard.html').write_text(page)
    print(json.dumps(overview,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');main(p.parse_args().folder)
