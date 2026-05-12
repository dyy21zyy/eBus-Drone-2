from __future__ import annotations
import json
from pathlib import Path

def build_report(data:dict, out_path:str):
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2),encoding='utf-8')
    return str(p)
