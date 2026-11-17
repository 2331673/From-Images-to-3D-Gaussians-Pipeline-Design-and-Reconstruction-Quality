#!/usr/bin/env python3
"""Compute per-object training gains and capacity ratios from saved metrics."""
from pathlib import Path
import json
from statistics import mean, median
R=Path(__file__).resolve().parents[1]
rows=[x for x in json.loads((R/'results/summary.json').read_text())['results'] if x['kind']=='primary']
rows.sort(key=lambda x:x['number'])
d=[x['curve']['6000']['psnr']-x['curve']['2000']['psnr'] for x in rows]
g=[x['curve']['6000']['num_GS']/x['curve']['2000']['num_GS'] for x in rows]
record=dict(source='results/summary.json',comparison='2000 to 6000 optimization steps, same runs',psnr_deltas=d,gaussian_count_ratios=g,psnr_delta_macro=mean(d),psnr_delta_median=median(d),psnr_delta_excluding_05=mean(d[:4]+d[5:]),object_05_share_of_net_psnr_increase=d[4]/sum(d),lpips_improved_objects=[x['number'] for x in rows if x['curve']['6000']['lpips']<x['curve']['2000']['lpips']])
(R/'results/paper_analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
