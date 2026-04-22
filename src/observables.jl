abstract type AbstractObservable end

function labels(obs::AbstractObservable, state_labels::AbstractVector{<:AbstractString})
    throw(MethodError(labels, (obs, state_labels)))
end

function labels(obs::AbstractObservable, n_states::Int)
    return labels(obs, _default_state_labels(n_states))
end

function label(obs::AbstractObservable, state_labels::AbstractVector{<:AbstractString}, idx::Int)
    obs_labels = labels(obs, state_labels)
    checkbounds(obs_labels, idx)
    return obs_labels[idx]
end

label(obs::AbstractObservable, n_states::Int, idx::Int) = label(obs, _default_state_labels(n_states), idx)

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
labels(o::Identity, state_labels::AbstractVector{<:AbstractString}) = String.(state_labels[o.idx])

_function_label(f) = string(f)
_delay_expression(expr::AbstractString, τ::Int) = τ == 0 ? expr : "delay($(expr), $(τ))"

struct BroadcastObs{F} <: AbstractObservable
    f::F
end
max_delay(::BroadcastObs) = 0
(o::BroadcastObs)(X::AbstractMatrix, t::AbstractVector{Int}) = o.f.(X[:, t])
labels(o::BroadcastObs, state_labels::AbstractVector{<:AbstractString}) = ["$(_function_label(o.f))($(state_label))" for state_label in state_labels]

struct MappedObs{F, O <: AbstractObservable} <: AbstractObservable
    f::F
    obs::O
end
max_delay(o::MappedObs) = max_delay(o.obs)
(o::MappedObs)(X::AbstractMatrix, t::AbstractVector{Int}) = o.f.(o.obs(X, t))
labels(o::MappedObs, state_labels::AbstractVector{<:AbstractString}) = ["$(_function_label(o.f))($(obs_label))" for obs_label in labels(o.obs, state_labels)]

Base.:∘(f::Function, obs::AbstractObservable) = MappedObs(f, obs)


struct ChainedObs{O1 <: AbstractObservable, O2 <: AbstractObservable} <: AbstractObservable
    outer::O1
    inner::O2
end
max_delay(o::ChainedObs) = max_delay(o.outer) + max_delay(o.inner)

function (o::ChainedObs)(X::AbstractMatrix, t::AbstractVector{Int})
    Y = o.inner(X, t)
    return o.outer(Y, 1:size(Y, 2))
end
labels(o::ChainedObs, state_labels::AbstractVector{<:AbstractString}) = labels(o.outer, labels(o.inner, state_labels))

Base.:∘(obs1::AbstractObservable, obs2::AbstractObservable) = ChainedObs(obs1, obs2)

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
labels(o::DelayObs{<:Any, Int}, state_labels::AbstractVector{<:AbstractString}) = [_delay_expression(obs_label, o.delay) for obs_label in labels(o.obs, state_labels)]

(o::DelayObs{<:Any, <:AbstractVector})(X::AbstractMatrix, t::AbstractVector{Int}) = 
    mapreduce(τ -> o.obs(X, t .- τ), vcat, o.delay)
function labels(o::DelayObs{<:Any, <:AbstractVector}, state_labels::AbstractVector{<:AbstractString})
    out = String[]
    obs_labels = labels(o.obs, state_labels)
    for τ in o.delay
        append!(out, [_delay_expression(obs_label, τ) for obs_label in obs_labels])
    end
    return out
end

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
function labels(o::StackedObs, state_labels::AbstractVector{<:AbstractString})
    out = String[]
    for obs in o.observables
        append!(out, labels(obs, state_labels))
    end
    return out
end

wrap_obs(f::Function) = BroadcastObs(f)
wrap_obs(o::AbstractObservable) = o
Base.vcat(args::Union{Function, AbstractObservable}...) = StackedObs(map(wrap_obs, args))

_state_count(::Colon) = nothing
_state_count(idx::Int) = idx
_state_count(idx::AbstractUnitRange{<:Integer}) = isempty(idx) ? 0 : last(idx)
_state_count(idx::AbstractVector{<:Integer}) = isempty(idx) ? 0 : maximum(idx)

_merge_state_counts(counts) = any(isnothing, counts) ? nothing : maximum(counts; init=0)

_label_state_count(::BroadcastObs) = nothing
_label_state_count(o::Identity) = _state_count(o.idx)
_label_state_count(o::MappedObs) = _label_state_count(o.obs)
_label_state_count(o::ChainedObs) = _label_state_count(o.inner)
_label_state_count(o::DelayObs) = _label_state_count(o.obs)
_label_state_count(o::StackedObs) = _merge_state_counts(_label_state_count.(o.observables))

function _show_entries(io::IO, entries::AbstractVector{<:AbstractString}, reserved_lines::Int)
    available_lines = max(displaysize(io)[1] - reserved_lines, 1)

    if length(entries) <= available_lines
        for entry in entries
            print(io, "\n ", entry)
        end
        return
    end

    visible_lines = max(available_lines - 1, 0)
    head = cld(visible_lines, 2)
    tail = fld(visible_lines, 2)

    for entry in entries[1:head]
        print(io, "\n ", entry)
    end

    print(io, "\n ⋮")

    for entry in entries[end-tail+1:end]
        print(io, "\n ", entry)
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs::AbstractObservable)
    n_states = _label_state_count(obs)
    delay = max_delay(obs)

    if isnothing(n_states)
        print(io, "lifting: ? -> ?")
        if delay > 0
            print(io, " (max delay: ", delay, ")")
        end
        print(io, "\n use `labels(obs, n_states)` to display expressions")
        return
    end

    obs_labels = labels(obs, n_states)
    print(io, "lifting: ", n_states, " -> ", length(obs_labels))
    if delay > 0
        print(io, " (max delay: ", delay, ")")
    end
    _show_entries(io, obs_labels, 1)
end
