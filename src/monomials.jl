function exponent_vector(n::Int, total_degree::Int)
    exps = Vector{Vector{Int}}()
    current = zeros(Int, n)
    function gen(i::Int, remaining::Int)
        if i == 1
            current[1] = remaining
            push!(exps, copy(current))
            return
        end
        for v in 0:remaining
            current[i] = v
            gen(i - 1, remaining - v)
        end
    end
    gen(n, total_degree)
    return exps
end

struct Monomials{O <: AbstractObservable} <: AbstractObservable
    obs::O
    degrees::Vector{Int}
    function Monomials(obs::O, degrees::Vector{Int}) where {O <: AbstractObservable}
        if isempty(degrees)
            throw(ArgumentError("degree list cannot be empty"))
        end
        if any(d -> d < 0, degrees)
            throw(ArgumentError("degrees must be nonnegative"))
        end
        return new{O}(obs, degrees)
    end
end

Monomials(obs::O, degrees::AbstractVector{<:Integer}) where {O <: AbstractObservable} = Monomials(obs, Int.(degrees))
Monomials(obs::O, degree::Integer) where {O <: AbstractObservable} = Monomials(obs, [Int(degree)])
Monomials(obs::O, degrees::AbstractUnitRange{<:Integer}) where {O <: AbstractObservable} = Monomials(obs, collect(Int, degrees))

Monomials(degree::Integer) = Monomials(Identity(), degree)
Monomials(degrees::AbstractUnitRange{<:Integer}) = Monomials(Identity(), degrees)
Monomials(degrees::AbstractVector{<:Integer}) = Monomials(Identity(), degrees)

Base.:∘(m::Monomials{<:Identity}, obs::AbstractObservable) = Monomials(obs, copy(m.degrees))

max_delay(m::Monomials) = max_delay(m.obs)
_label_state_count(m::Monomials) = _label_state_count(m.obs)

function _degrees_repr(degrees::Vector{Int})
    length(degrees) == 1 && return string(only(degrees))
    return repr(degrees)
end

function _observable_repr(m::Monomials)
    monomial_repr = "Monomials($(_degrees_repr(m.degrees)))"
    if m.obs isa Identity && m.obs.idx isa Colon
        return monomial_repr
    end
    return monomial_repr * " ∘ " * _observable_repr(m.obs)
end

function _power_expression(expr::AbstractString, exponent::Int)
    exponent == 1 && return expr
    return "($(expr))^$(exponent)"
end

function _monomial_expression(obs_labels::Vector{String}, exponents::Vector{Int})
    factors = String[]
    for (obs_label, exponent) in zip(obs_labels, exponents)
        exponent == 0 && continue
        push!(factors, _power_expression(obs_label, exponent))
    end

    isempty(factors) && return "1"
    return join(factors, " * ")
end

function labels(m::Monomials, state_labels::AbstractVector{<:AbstractString})
    obs_labels = labels(m.obs, state_labels)
    exponents = monomial_exponents(length(obs_labels), m.degrees)
    return [_monomial_expression(obs_labels, exponent) for exponent in exponents]
end

function monomial_exponents(n::Int, degrees::Vector{Int})
    exps = Vector{Vector{Int}}()
    for total_degree in degrees
        append!(exps, exponent_vector(n, total_degree))
    end
    return exps
end


function (m::Monomials)(X::AbstractMatrix, t::AbstractVector{Int})
    # Evaluate whatever observable is *inside* the monomials layer
    Y = m.obs(X, t)
    n, T = size(Y)
    
    # Generate exponent vectors dynamically based on inner dimension 'n'
    exps = monomial_exponents(n, m.degrees)
    
    # Preallocate the lifted matrix
    out = zeros(eltype(Y), length(exps), T)
    
    # Evaluate column-by-column (fast, type-stable loop)
    for (i, e) in enumerate(exps)
        for j in 1:T
            val = one(eltype(Y))
            for k in 1:n
                # ^0 correctly evaluates to 1 in Julia
                val *= Y[k, j]^e[k] 
            end
            out[i, j] = val
        end
    end
    
    return out
end
