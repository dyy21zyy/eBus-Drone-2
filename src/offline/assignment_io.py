from pathlib import Path
import json

def write_assignment(result:dict,path:str)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2))
