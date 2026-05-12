from pathlib import Path
import json

def write_instance(data:dict, out_dir:str, name:str)->Path:
    p=Path(out_dir); p.mkdir(parents=True,exist_ok=True)
    fp=p/f"{name}.json"; fp.write_text(json.dumps(data,indent=2))
    return fp
