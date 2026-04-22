function _default_state_labels(n_states::Int)
    n_states < 1 && throw(ArgumentError("n_states must be positive"))
    return ["x[$i]" for i in 1:n_states]
end
