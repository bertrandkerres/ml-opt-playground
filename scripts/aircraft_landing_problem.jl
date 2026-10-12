#= Aircraft landing problem
From text2zonc dataset, problem 3
### Problem description
The Aircraft Landing Problem involves determining optimal landing times for a set of aircraft while minimizing penalties associated with deviations from target landing times. Each aircraft must land within a given time window and maintain a required separation time from other aircraft. Landing too early or too late incurs a penalty proportional to the time difference. The objective is to minimize the total penalty while ensuring all constraints are met.

### Parameters
 * TotalAircrafts: The total number of aircraft in the set
 * EarliestLandingTime: The earliest landing time for each aircraft
 * LatestLandingTime: The latest landing time for each aircraft
 * TargetLandingTime: The target landing time for each aircraft
 * PenaltyTimeAfterTarget: The penalty incurred for each aircraft per unit landing after the target time
 * PenaltyTimeBeforeTarget: The penalty incurred for each aircraft per unit landing before the target time
 * SeparationTimeMatrix: The required separation time between the landing of aircraft i and aircraft j

### Parameter values
The parameter values are given in minizinc format:
```code
TotalAircrafts = 10;

EarliestLandingTime = [129, 195, 89, 90, 110, 120, 124, 126, 135, 160];
TargetLandingTime = [155, 258, 98, 106, 123, 135, 138, 140, 150, 180];
LatestLandingTime = [689, 653, 517, 501, 634, 603, 657, 592, 510, 604];

PenaltyTimeAfterTarget = [24, 25, 10, 13, 10, 20, 24, 12, 16, 27];
PenaltyTimeBeforeTarget = [24, 25, 10, 13, 10, 20, 24, 12, 16, 27];

SeparationTimeMatrix = array2d(1..10, 1..10,
  [
    99999, 11, 12, 10, 10, 11, 12, 12, 12, 10,
    14, 99999, 10, 12, 12, 10, 13, 14, 11, 13,
    11, 14, 99999, 10, 11, 12, 9, 10, 11, 13,
    8, 10, 11, 99999, 8, 12, 8, 8, 9, 9,
    10, 10, 14, 14, 99999, 10, 8, 14, 11, 10,
    11, 9, 11, 11, 14, 99999, 9, 9, 9, 12,
    12, 13, 13, 8, 14, 14, 99999, 8, 13, 11,
    14, 8, 8, 14, 12, 8, 14, 99999, 8, 12,
    11, 12, 11, 11, 13, 11, 11, 14, 99999, 9,
    11, 9, 10, 10, 8, 14, 8, 14, 9, 99999
  ]
);
```
=#

using LinearAlgebra
using JuMP
import HiGHS

TotalAircrafts = 10

EarliestLandingTime = [129, 195, 89, 90, 110, 120, 124, 126, 135, 160]
TargetLandingTime = [155, 258, 98, 106, 123, 135, 138, 140, 150, 180]
LatestLandingTime = [689, 653, 517, 501, 634, 603, 657, 592, 510, 604]

PenaltyTimeAfterTarget = [24, 25, 10, 13, 10, 20, 24, 12, 16, 27]
PenaltyTimeBeforeTarget = [24, 25, 10, 13, 10, 20, 24, 12, 16, 27]

SeparationTimeMatrix = [
    99999 11 12 10 10 11 12 12 12 10;
    14 99999 10 12 12 10 13 14 11 13;
    11 14 99999 10 11 12 9 10 11 13;
    8 10 11 99999 8 12 8 8 9 9;
    10 10 14 14 99999 10 8 14 11 10;
    11 9 11 11 14 99999 9 9 9 12;
    12 13 13 8 14 14 99999 8 13 11;
    14 8 8 14 12 8 14 99999 8 12;
    11 12 11 11 13 11 11 14 99999 9;
    11 9 10 10 8 14 8 14 9 99999;
]

mdl = Model(HiGHS.Optimizer)

@variable(mdl, x[1:10] >= 0)
@constraint(mdl, time_window[i=1:10], EarliestLandingTime[i] <= x[i] <= LatestLandingTime[i])

@variable(mdl, d_early[1:10])
@variable(mdl, d_late[1:10])
@constraint(mdl, early_deviation[i=1:10], d_early[i] == TargetLandingTime[i] - x[i])
@constraint(mdl, late_deviation[i=1:10], d_late[i] == x[i] - TargetLandingTime[i])

@variable(mdl, t_early[1:10] >= 0)
@constraint(mdl, early_epigraph[i=1:10], t_early[i] >= d_early[i])
@variable(mdl, t_late[1:10] >= 0)
@constraint(mdl, late_epigraph[i=1:10], t_late[i] >= d_late[i])

@objective(mdl, Min, dot(t_early, PenaltyTimeBeforeTarget) + dot(t_late, PenaltyTimeAfterTarget))


@variable(mdl, dx[1:10, 1:10])
@constraint(mdl, separation[i=1:10, j=1:10], dx[i, j] == x[i] - x[j])

M = maximum(LatestLandingTime)
# delta_ij == 1 if i lands before j, 0 else
@variable(mdl, delta_ij[1:10, 1:10], Bin)
# One has to land before the other
@constraint(mdl, delta_ij_symmetry[i=1:10, j=1:10; i < j], delta_ij[i, j] + delta_ij[j, i] == 1)
# i lands after j -> delta_ij = 0 -> dx[i, j] >= Separation[first j, then i]
@constraint(mdl, separation_i_after_j[i=1:10, j=1:10; i != j], dx[i, j] + M * delta_ij[i, j] >= SeparationTimeMatrix[j, i])
# i lands before j -> delta_ij = 1 -> dx[i, j] >= Separation[first i, then j]
@constraint(mdl, separation_j_after_i[i=1:10, j=1:10; i != j], dx[j, i] - M * delta_ij[i, j] >= SeparationTimeMatrix[i, j] - M)

optimize!(mdl)

println(solution_summary(mdl))