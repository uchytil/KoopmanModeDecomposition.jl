abstract type AbstractObservable end

function (obs::AbstractObservable)(X::AbstractMatrix)
    d = max_delay(obs)
    t = (d + 1):size(X, 2)
    return obs(X, t)
end

struct Identity{I} <: AbstractObservable
    idx::I
end
Identity() = Identity(Colon())
Identity(idx::Int) = Identity(idx:idx)
max_delay(::Identity) = 0
(o::Identity)(X::AbstractMatrix, t::AbstractVector{Int}) = X[o.idx, t]

struct BroadcastObs{F} <: AbstractObservable
    f::F
end
max_delay(::BroadcastObs) = 0
(o::BroadcastObs)(X::AbstractMatrix, t::AbstractVector{Int}) = o.f.(X[:, t])

struct MappedObs{F, O <: AbstractObservable} <: AbstractObservable
    f::F
    obs::O
end
max_delay(o::MappedObs) = max_delay(o.obs)
(o::MappedObs)(X::AbstractMatrix, t::AbstractVector{Int}) = o.f.(o.obs(X, t))

Base.:∘(f::Function, obs::AbstractObservable) = MappedObs(f, obs)

# Let D be unrestricted so it can hold Int, UnitRange, or Vector
struct Delay{D}
    τ::D
end

struct DelayedFunction{F, D}
    f::F
    delay::D
end

struct DelayObs{O <: AbstractObservable, D} <: AbstractObservable
    obs::O
    delay::D
end

_max_d(d::Int) = d
_max_d(d::AbstractVector) = maximum(d)
max_delay(o::DelayObs) = _max_d(o.delay) + max_delay(o.obs)

(o::DelayObs{<:Any, Int})(X::AbstractMatrix, t::AbstractVector{Int}) = o.obs(X, t .- o.delay)

(o::DelayObs{<:Any, <:AbstractVector})(X::AbstractMatrix, t::AbstractVector{Int}) = 
    mapreduce(τ -> o.obs(X, t .- τ), vcat, o.delay)

_combine_delays(d1::Int, d2::Int) = d1 + d2
_combine_delays(d1::Int, d2::AbstractVector) = d1 .+ d2
_combine_delays(d1::AbstractVector, d2::Int) = d1 .+ d2
_combine_delays(d1::AbstractVector, d2::AbstractVector) = vec(d1 .+ d2')

Base.:∘(d1::Delay, d2::Delay) = Delay(_combine_delays(d1.τ, d2.τ))
Base.:∘(d::Delay, f::Function) = DelayedFunction(f, d.τ)
Base.:∘(f::Function, d::Delay) = DelayedFunction(f, d.τ)
Base.:∘(df::DelayedFunction, d::Delay) = DelayedFunction(df.f, _combine_delays(df.delay, d.τ))
Base.:∘(d::Delay, df::DelayedFunction) = DelayedFunction(df.f, _combine_delays(d.τ, df.delay))
Base.:∘(df::DelayedFunction, f::Function) = DelayedFunction(df.f ∘ f, df.delay)
Base.:∘(f::Function, df::DelayedFunction) = DelayedFunction(f ∘ df.f, df.delay)
Base.:∘(df1::DelayedFunction, df2::DelayedFunction) = DelayedFunction(df1.f ∘ df2.f, _combine_delays(df1.delay, df2.delay))
Base.:∘(d::Delay, obs::AbstractObservable) = DelayObs(obs, d.τ)
Base.:∘(df::DelayedFunction, obs::AbstractObservable) = DelayObs(df.f ∘ obs, df.delay)

struct StackedObs{T <: Tuple} <: AbstractObservable
    observables::T
end
max_delay(o::StackedObs) = maximum(max_delay.(o.observables))
(o::StackedObs)(X::AbstractMatrix, t::AbstractVector{Int}) = mapreduce(obs -> obs(X, t), vcat, o.observables)

wrap_obs(f::Function) = BroadcastObs(f)
wrap_obs(o::AbstractObservable) = o
Base.vcat(args::Union{Function, AbstractObservable}...) = StackedObs(map(wrap_obs, args))