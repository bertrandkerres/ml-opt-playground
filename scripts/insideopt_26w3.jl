#=
Along a 1000 meter long beach, you and two more ice cream vendors are setting up shop.
You arrive first. Where do you place your cart?

Ice Cream Vendor etiquette dictates that no two vendors set up their carts closer than
10 meters from one another. Moreover, once you set up your shop, you must not move it
for the rest of the day. Every part of the beach has the same density of customers, but
they are lazy and will frequent only the ice cream stand closest to them. 

Assuming that the other two vendors will arrive one after the other and that they will
choose a location that will maximize their income, where should you set up your cart?
=#
import Plots

const L = 1000

function profit(me, others)
    x1, x2 = others
    # Make sure they're ordered to limit cases to check
    if x1 > x2
        x1, x2 = x2, x1
    end

    if me < 0 || me > L || is_too_close(me, x1) || is_too_close(me, x2)
        # Invalid domain
        return NaN
    end

    if me > x2
        # Rightmost
        return (me - x2) / 2 + (L - me)
    elseif me > x1 # && me < x2
        # In between
        # (me - x1) / 2 + (x2 - me) / 2
        return (x2 - x1) / 2
    else
        # Leftmost
        return me + (x1 - me) / 2
    end
end

is_too_close(x1, x2) = abs(x2 - x1) <= 9.99


function choose_x2(x1)
    # min x2 s.t. L-x2 <= x1 OR L-x2 <= (x2-x1)/2
    x2_lower = min(L - x1, (2L + x1) / 3)
    return x2_lower + 10.0
end

function choose_x3(x1, x2)
    # Returns lower, upper for uniform dist
    i = argmax((x1, L - x2, (x2 - x1) / 2))
    if i == 1
        return (x1 - 10.0, x1 - 10.0)
    elseif i == 2
        return (x2 + 10.0, x2 + 10.0)
    else
        return (x1 + 10.0, x2 - 10.0)
    end
end


function grid_search(; step = 1.0)
    L_symm = L ÷ 2
    X1 = 0:step:L_symm
    P_me = zeros(3, length(X1))
    pos = zeros(3 + 2, length(X1))
    for (i, x1) in enumerate(X1)
        x2 = choose_x2(x1)
        x3l, x3u = choose_x3(x1, x2)
        x3m = (x3l + x3u) / 2
        pos[:, i] .= [x1; x2; x3m; x3l; x3u]
        pm, pl, pu = profit(x1, (x2, x3m)), profit(x1, (x2, x3l)), profit(x1, (x2, x3u))
        P_me[:, i] = [pm; pl; pu]
    end
    return P_me, pos
end

P, pos = grid_search(step = 0.1)
imax = argmax(P[1, :])
x1 = pos[1, imax]
println("Solution: x1 = $x1")
println("Expected profit: $(P[:, imax])")

plt1 = Plots.plot(
    pos[1, :], pos[1:3, :]',
    label = ["A" "B" "E(C)"],
    ylim = [0, 1000],
    title = "Positions",
    ylabel = "Position [m]",
    linewidth = 2.0,
)

Plots.plot!(
    plt1, 
    pos[1, :], pos[4, :], fillrange = pos[5, :], fillalpha = 0.4, c = 3,
    linewidth = 0.0, label = "CI(C)"
)

plt2 = Plots.plot(
    pos[1, :], P[1, :];
    ylim = [0, 500],
    title = "Profit",
    xlabel = "A [m]",
    ylabel = "Exclusive beachfront [m]",
    linewidth = 2.0,
    label = "Expected"
)
Plots.plot!(
    plt2,
    pos[1, :], P[2, :], fillrange = P[3, :], fillalpha = 0.4, c = 1,
    linewidth = 0.0, label = "CI"
)
plt_combined = Plots.plot(plt1, plt2, layout = (2, 1), size = (600, 600))
