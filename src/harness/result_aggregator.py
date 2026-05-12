from __future__ import annotations
import math

def aggregate(items:list[dict])->dict:
    if not items: return {}
    keys=[k for k,v in items[0].items() if isinstance(v,(int,float))]
    out={"count":len(items)}
    for k in keys:
        vals=[float(i.get(k,0.0)) for i in items]
        mean=sum(vals)/len(vals)
        var=sum((x-mean)**2 for x in vals)/len(vals)
        out[k]={"mean":mean,"std":math.sqrt(var),"best":min(vals),"worst":max(vals),"count":len(vals)}
    return out
