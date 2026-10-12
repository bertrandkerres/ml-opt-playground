#=
Max Z
s.t.
  Z = min sin(a) * x1 + cos(a) * x2
  s.t.
    x1 + x2 = 1
    1/4 <= cos(a) * x1 + sin(a) * x2 <= 3/4
    x1, x2 in [0, 1]
  a in [0, 2pi]
=#

import HiGHS
using JuMP


a = 0:0.0001:2π

function inner_lp_vertices(a)
    sa, ca = sincos(a)
    da = ca - sa
    x1 = (0.25 - sa) / da
    x2 = (0.75 - sa) / da
    Z1 = sa * x1 + ca * (1 - x1)
    Z2 = sa * x2 + ca * (1 - x2)
    return Z1, Z2, sa, ca
end

Z = inner_lp_vertices.(a)
Zmin = map(z -> min(z...), Z)

using Plots
Plots.plot(a, Zmin)

i_max = argmax(Z)
a_max = a[i_max]