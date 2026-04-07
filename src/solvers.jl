abstract type AbstractKoopmanSolver end

struct PseudoInverse <: AbstractKoopmanSolver end

function fit(::PseudoInverse, Y::AbstractMatrix, Y_next::AbstractMatrix)
    return Y_next * pinv(Y)
end

struct TruncatedSVD <: AbstractKoopmanSolver
    atol::Float64
    rank::Union{Int,Nothing}
end
TruncatedSVD(; atol=1e-8, rank=nothing) = TruncatedSVD(atol, rank)

function fit(solver::TruncatedSVD, Y::AbstractMatrix, Y_next::AbstractMatrix)
    F = svd(Y)

    r = if isnothing(solver.rank)
        count(>(solver.atol), F.S)
    else
        min(solver.rank, length(F.S))
    end

    U_r = @view F.U[:, 1:r]
    S_r_inv = Diagonal(1 ./ F.S[1:r])
    Vt_r = @view F.Vt[1:r, :]

    return Y_next * (Vt_r' * S_r_inv * U_r')
end

function fit(solver::AbstractKoopmanSolver, dict::AbstractObservable, X::AbstractMatrix)
    Y_full = dict(X)

    Y = @view Y_full[:, 1:end-1]
    Y_next = @view Y_full[:, 2:end]

    K = fit(solver, Y, Y_next)

    return KoopmanModel(dict, K)
end


function fit(solver::AbstractKoopmanSolver, dict::AbstractObservable, Xs::Vector{<:AbstractMatrix})

    Y = AbstractMatrix[]
    Y_next = AbstractMatrix[]

    for X in Xs
        Y_full = dict(X)
        
        push!(Y, @view Y_full[:, 1:end-1])
        push!(Y_next, @view Y_full[:, 2:end])
    end

    Y = reduce(hcat, Y)
    Y_next = reduce(hcat, Y_next)

    K = fit(solver, Y, Y_next)

    return KoopmanModel(dict, K)
end