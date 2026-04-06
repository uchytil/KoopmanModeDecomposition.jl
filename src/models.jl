struct KoopmanModel{O <: AbstractObservable, M <: AbstractMatrix}
    dict::O
    operator::M
end

function predict(model::KoopmanModel, X::AbstractMatrix, steps::Int)
    d = max_delay(model.dict)
    if size(X, 2) < d + 1
        error("Initial condition matrix X must have at least $(d + 1) columns to satisfy the maximum delay of $d.")
    end
    
    Y = model.dict(X)
    n_dims, t_start = size(Y)
    
    Y_traj = zeros(eltype(Y), n_dims, t_start + steps)
    
    Y_traj[:, 1:t_start] .= Y
    
    y = Y[:, end]
    for i in 1:steps
        y = model.operator * y
        Y_traj[:, t_start + i] .= y
    end
    
    return Y_traj
end

function predict(model::KoopmanModel, x::AbstractVector, steps::Int)
    return predict(model, reshape(x, :, 1), steps)
end