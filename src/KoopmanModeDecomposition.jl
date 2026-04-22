module KoopmanModeDecomposition

using LinearAlgebra

export AbstractObservable, Identity, Delay, StackedObs
export Monomials
export labels, label

export KoopmanModel, predict
export ModalDecomposition, decompose, eigenfunctions

export AbstractKoopmanSolver, PseudoInverse, TruncatedSVD, fit

include("labels.jl")
include("observables.jl")
include("monomials.jl")
include("models.jl")       # Must be loaded before solver so KoopmanModel exists
include("solvers.jl")       # Can now safely return a KoopmanModel
include("decomposition.jl")

end
