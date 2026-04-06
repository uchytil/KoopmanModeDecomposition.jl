module KoopmanModeDecomposition

using LinearAlgebra

export AbstractObservable, Identity, Delay, StackedObs
export Monomials

export KoopmanModel, predict

export AbstractKoopmanSolver, PseudoInverse, TruncatedSVD, fit

include("observables.jl")
include("monomials.jl")
include("models.jl")       # Must be loaded before solver so KoopmanModel exists
include("solvers.jl")       # Can now safely return a KoopmanModel

end
