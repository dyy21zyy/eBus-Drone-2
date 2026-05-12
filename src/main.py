import argparse
from pathlib import Path
from src.utils.config import load_yaml
from src.utils.logger import log_line
from src.data_generation.scenario_generator import generate_scenario
from src.data_generation.instance_writer import write_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.offline.assignment_io import write_assignment
from src.harness.experiment_runner import run_experiment


def run_generate(cfg, instance, seed):
    num=load_yaml(f"configs/instances/{instance}.yaml").get("num_stations",10)
    s=generate_scenario(num,seed)
    out=write_instance(s,f"data/generated/{instance}",f"seed_{seed}")
    log_line("outputs/logs/generate.log",f"generated {out}")

def run_offline(instance):
    import json
    p=Path(f"data/generated/{instance}")
    first=sorted(p.glob("*.json"))[0]
    sc=json.loads(first.read_text())
    result=solve_assignment(build_assignment_data(sc))
    write_assignment(result,f"outputs/assignments/{instance}_assignment.json")
    log_line("outputs/logs/offline.log","offline completed")

def run_mode(mode,cfg,instance,seed):
    if mode in ["generate","all"]: run_generate(cfg,instance,seed)
    if mode in ["offline","all"]: run_offline(instance)
    if mode in ["train","eval","benchmark","ablation","sensitivity","all"]:
        run_experiment(mode if mode!='all' else 'all',"outputs",{"mode":mode,"seed":seed})
        log_line("outputs/logs/pipeline.log",f"ran {mode}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['generate','offline','train','eval','benchmark','ablation','sensitivity','all'])
    ap.add_argument('--config',required=True)
    ap.add_argument('--instance',default='small')
    ap.add_argument('--seed',type=int,default=1)
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    run_mode(args.mode,cfg,args.instance,args.seed)

if __name__=='__main__':
    main()
