struct FastNode{T} <: OnlineStat{(1, 0)}
    stats::Matrix{T}
    id::Int 
    children::Vector{Int}
    j::Int 
    at::Float64
    ig::Float64
end