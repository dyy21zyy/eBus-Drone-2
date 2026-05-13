from src.low_level.station_operator import operate_station_step
from src.env.ebus_drone_env import EBusDroneEnv


def test_dispatch_consumes_one_full_battery():
    st={"station_id":1,"drones":[{"drone_id":"d1","status":"idle"}],"locker_parcels":[1],"full_batteries":1,"depleted_batteries":0,"empty_batteries":0,"G_max":1,"charging_batteries":[],"P_capacity":100,"P_bat":5}
    parcels={1:{"id":1,"parcel_id":1,"status":"in_locker","assigned_station_id":1,"release_time_min":0.0,"T_out_min":2,"T_rt_min":4,"deadline_min":100,"drone_cost":1.0,"weight_kg":1.0}}
    operate_station_step(st,0.0,parcel_states=parcels,p_e=0.0,p_l=0.0,new_parcels=True)
    assert st["full_batteries"]==0


def test_return_creates_depleted_battery():
    st={"station_id":1,"drones":[{"drone_id":"d1","status":"active","return_time":3.0}],"locker_parcels":[],"full_batteries":0,"depleted_batteries":0,"empty_batteries":0,"G_max":0,"charging_batteries":[],"P_capacity":100,"P_bat":5}
    operate_station_step(st,3.0,parcel_states={},p_e=0.0,p_l=0.0,new_parcels=False)
    assert st["depleted_batteries"]==1


def test_charge_completion_moves_empty_to_full():
    st={"station_id":1,"drones":[],"locker_parcels":[],"full_batteries":0,"depleted_batteries":1,"empty_batteries":1,"G_max":1,"charging_batteries":[],"P_capacity":100,"P_bat":10,"battery_charge_duration_min":2}
    operate_station_step(st,0.0,parcel_states={},p_e=0.0,p_l=0.0,new_parcels=False)
    assert st["depleted_batteries"]==0
    operate_station_step(st,2.0,parcel_states={},p_e=0.0,p_l=0.0,new_parcels=False)
    assert st["full_batteries"]==1


def test_max_round_trip_duration_filters():
    st={"station_id":1,"drones":[{"drone_id":"d1","status":"idle"}],"locker_parcels":[1],"full_batteries":1,"depleted_batteries":0,"empty_batteries":0,"G_max":1,"charging_batteries":[],"P_capacity":100,"P_bat":5}
    parcels={1:{"id":1,"parcel_id":1,"status":"in_locker","assigned_station_id":1,"release_time_min":0.0,"T_out_min":10,"T_rt_min":200,"deadline_min":100,"drone_cost":1.0,"weight_kg":1.0}}
    out=operate_station_step(st,0.0,parcel_states=parcels,p_e=0.0,p_l=0.0,new_parcels=True,max_round_trip_duration=120.0)
    assert out["n_disp"]==0


def test_station_progress_between_decisions():
    env=EBusDroneEnv(smoke_test=True)
    sid=1
    st=env.station_states[sid]
    st["depleted_batteries"]=1
    st["empty_batteries"]=1
    st["battery_charge_duration_min"]=1
    env._run_station_interval(0.0,2.0)
    assert st["full_batteries"]>=2

def test_bus_charging_energy_positive_during_interval():
    env=EBusDroneEnv(smoke_test=True)
    st=env.station_states[1]
    st["charger_release_times_min"]= [10.0]
    env._run_station_interval(0.0,5.0)
    assert st.get("bus_charging_energy_kwh",0.0) > 0.0


def test_overload_and_locker_overflow_accumulate_over_time():
    env=EBusDroneEnv(smoke_test=True)
    st=env.station_states[1]
    st["P_capacity"]=10.0
    st["base_load_fallback_kw"]=20.0
    st["charger_release_times_min"]=[10.0]
    env.parcel_states[99]={"parcel_id":99,"id":99,"status":"in_locker","assigned_station_id":1,"weight_kg":150.0}
    st["locker_parcels"]=[99]
    st["locker_capacity_kg"]=100.0
    env._run_station_interval(0.0,2.0)
    assert st.get("power_overload_amount_kw_min",0.0) > 0.0
    assert st.get("power_overload_duration_min",0.0) > 0.0
    assert st.get("locker_overflow_amount_kg_min",0.0) > 0.0
