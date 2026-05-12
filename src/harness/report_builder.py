from pathlib import Path

def write_report(path:str,text:str)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
