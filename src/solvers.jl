abstract type AbstractKoopmanSolver end

function fit_projection(dict::AbstractObservable, X::AbstractMatrix)
    Y = dict(X)
    d = max_delay(dict)
    X_target = @view X[:, d + 1:end]
    return X_target * pinv(Y)
end

function fit_projection(dict::AbstractObservable, Xs::Vector{<:AbstractMatrix})
    Y = AbstractMatrix[]
    X_target = AbstractMatrix[]

    d = max_delay(dict)
    for X in Xs
        push!(Y, dict(X))
        push!(X_target, @view X[:, d + 1:end])
    end

    return reduce(hcat, X_target) * pinv(reduce(hcat, Y))
end

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
    projection = fit_projection(dict, X)

    return KoopmanModel(dict, K, projection)
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
    projection = fit_projection(dict, Xs)

    return KoopmanModel(dict, K, projection)
end
