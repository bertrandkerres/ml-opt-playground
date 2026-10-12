using ControlSystemsBase

function bb2gb(sys; Ts = nothing)
    Pc = d2c(sys)
    Cinv = inv(Pc.C)
    Pc_gb = similarity_transform(Pc, Cinv)
    Ts === nothing && return Pc_gb
    return c2d(Pc_gb, Ts)
end


function build_rc_ss(p::AbstractVector, nx::Int, nu::Int, Ts::Real)
    if nx > 1
        al = (1:nx-1)
        a  = (1:nx) .+ al[end]
        au = (1:nx-1) .+ a[end]
        A = Matrix(Tridiagonal(p[al], p[a], p[au]))
    else
        A = [p[1];;]
        au = 1:1
    end
    bi = (1:nx*nu) .+ au[end]
    B = reshape(p[bi], nx, nu)
    C = I
    D = zeros(nx, nu)
    return ss(A, B, C, D, Ts)
end

trivec(A) = [A[diagind(A, -1)]; A[diagind(A, 0)]; A[diagind(A, 1)]]