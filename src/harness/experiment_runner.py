from pathlib import Path
import json

def run_experiment(name:str, outputs:str, payload:dict)->str:
    p=Path(outputs)/"metrics"/f"{name}.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2)); return str(p)
