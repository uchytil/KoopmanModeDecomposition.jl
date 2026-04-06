function exponent_vectors(n::Int, α::Int; drop_constant::Bool = false)
    exps = Vector{Vector{Int}}()
    for total in 0:α
        if drop_constant && total == 0
            continue
        end
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
        gen(n, total)
    end
    return exps
end

struct Monomials{O <: AbstractObservable} <: AbstractObservable
    obs::O
    α::Int
    drop_constant::Bool
end

Monomials(α::Int; drop_constant::Bool=false) = Monomials(Identity(), α, drop_constant)

Base.:∘(m::Monomials{<:Identity}, obs::AbstractObservable) = Monomials(obs, m.α, m.drop_constant)

max_delay(m::Monomials) = max_delay(m.obs)


function (m::Monomials)(X::AbstractMatrix, t::AbstractVector{Int})
    # Evaluate whatever observable is *inside* the monomials layer
    Y = m.obs(X, t)
    n, T = size(Y)
    
    # Generate exponent vectors dynamically based on inner dimension 'n'
    exps = exponent_vectors(n, m.α; drop_constant = m.drop_constant)
    
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