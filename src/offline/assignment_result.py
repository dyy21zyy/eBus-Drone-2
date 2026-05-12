def summarize_assignment(result:dict)->dict:
    return {"n":len(result.get("pairs",[])),"cost":result.get("cost",0.0)}
