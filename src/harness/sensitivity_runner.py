from __future__ import annotations
import csv, json, time
from copy import deepcopy
from pathlib import Path
from src.harness.benchmark_runner import build_policy
from src.harness.evaluator import evaluate_policy
from src.harness.result_aggregator import aggregate

FACTOR_PATHS={
    'passenger_intensity':('passenger','demand_intensity_factor', True),
    'num_customers':('generation','num_customers_factor', True),
    'parcel_intensity':('parcel','demand_intensity_factor', True),
    'chargers_per_station':('charging','chargers_per_station', True),
    'drones_per_station':('drone','drones_per_station', True),
    'locker_capacity':('parcel','locker_capacity_kg', True),
    'station_power_capacity':('power','station_capacity_kw', True),
    'initial_full_batteries':('battery','initial_fully_charged_per_station', False),
}

def run_sensitivity(methods, out_csv:str, env_builder, instance_name:str, test_seeds:list[int], cfg:dict, factor:str, values:list[float], smoke_test: bool=False, train_if_missing: bool=False):
    if factor not in FACTOR_PATHS: raise ValueError(f'Unsupported sensitivity factor: {factor}')
    rows=[]
    k1,k2,regen=FACTOR_PATHS[factor]
    for v in values:
        cfg_mod=deepcopy(cfg); cfg_mod.setdefault(k1,{})[k2]=v
        for seed in test_seeds:
            for m in methods:
                env=env_builder(seed, cfg_mod)
                t0=time.time()
                pol=build_policy(m, env, out_root=cfg['paths']['outputs'], train_if_missing=train_if_missing, smoke_test=smoke_test, cfg=cfg_mod, seed=seed, instance_name=instance_name)
                met=evaluate_policy(env, pol, episodes=1, max_steps=10 if smoke_test else None)
                met.update({'method':m,'instance':instance_name,'seed':seed,'factor':factor,'value':v,'runtime_sec':time.time()-t0,'requires_regeneration':regen,'smoke_mode':bool(smoke_test)})
                rows.append(met)
    if not rows: raise ValueError('Sensitivity produced no rows.')
    p=Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    grouped={}
    for m in methods:
        for v in values:
            grouped[f'{m}:{v}']=aggregate([r for r in rows if r['method']==m and r['value']==v])
    Path(out_csv).with_suffix('.json').write_text(json.dumps({'aggregated':grouped}, indent=2), encoding='utf-8')
    return rows
