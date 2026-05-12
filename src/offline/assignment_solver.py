def solve_assignment(data:list[tuple[int,int]])->dict:
    cost=float(sum(v for _,v in data))
    return {"pairs":data,"cost":cost}
