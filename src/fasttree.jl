struct FastNode{T} <: OnlineStat{(1, 0)}
    stats::Matrix{T}
    id::Int 
    children::Vector{Int}
    j::Int 
    at::Float64
    ig::Float64
end
function FastNode(p=0, nkeys=2; stat=FitNormal())
    FastNode([FitNormal() for i in 1:p, j in 1:nkeys], 1, Int[], 0, -Inf, -Inf)
end
function FastNode(o::FastNode, id::Int)
    FastNode([FitNormal() for i in 1:nvars(o), j in 1:nkeys(o)], id, Int[], 0, -Inf, -Inf)
end
nkeys(o::FastNode) = size(o.stats, 2)
nvars(o::FastNode) = size(o.stats, 1)
