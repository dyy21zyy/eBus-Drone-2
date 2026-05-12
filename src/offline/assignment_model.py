from __future__ import annotations

from dataclasses import dataclass
from src.offline.assignment_data_builder import AssignmentData


@dataclass
class OfflineAssignmentModel:
    data: AssignmentData
    variable_keys: list[tuple[int, int, int]]
    objective: list[float]


def build_assignment_model(data: AssignmentData) -> OfflineAssignmentModel:
    keys: list[tuple[int, int, int]] = []
    objective: list[float] = []
    for b in data.trips:
        for h in data.stations:
            for i in data.feasible_customers_by_station[h]:
                key = (b, h, i)
                keys.append(key)
                objective.append(
                    data.c_b[(b, h)] * data.parcel_weight[i]
                    + data.c_d[(h, i)]
                    + data.beta_h * data.parcel_weight[i] * data.h_bhi_0[key]
                    + data.beta_l * data.lateness_0_plus[key]
                )
    return OfflineAssignmentModel(data=data, variable_keys=keys, objective=objective)
