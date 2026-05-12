def valid_action_mask(waiting:int)->list[int]:
    return [1,1] if waiting>0 else [1,0]
