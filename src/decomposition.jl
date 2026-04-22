struct ModalDecomposition{L <: AbstractVector, V <: AbstractMatrix, W1 <: AbstractMatrix, W2 <: AbstractMatrix, D}
    eigenvalues::L
    modes::V
    right_eigenvectors::W1
    left_eigenvectors::W2
    projection::D
end

function _left_eigenvectors(right_vectors::AbstractMatrix)
    return adjoint(right_vectors \ I)
end

function decompose(model::KoopmanModel)
    isnothing(model.projection) && error("Koopman modes require a projection, but this model does not store one.")

    right = eigen(model.operator)
    left = _left_eigenvectors(right.vectors)
    modes = model.projection * right.vectors

    return ModalDecomposition(right.values, modes, right.vectors, left, model.projection)
end

function eigenfunctions(decomposition::ModalDecomposition, lifted::AbstractMatrix)
    return adjoint(decomposition.left_eigenvectors) * lifted
end

function eigenfunctions(decomposition::ModalDecomposition, model::KoopmanModel, X::AbstractMatrix)
    lifted = model.dict(X)
    return eigenfunctions(decomposition, lifted)
end

function eigenfunctions(decomposition::ModalDecomposition, model::KoopmanModel, x::AbstractVector)
    return eigenfunctions(decomposition, model, reshape(x, :, 1))
end
