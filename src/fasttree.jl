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

function fakedata(::Type{FastNode}, n, p) 
    x = randn(n, p)
    y = [(rand() > 1 /(1 + exp(xb))) + 1 for xb in x * (1:p)]
    x, y
end

function _fit!(o::FastNode, xy)
    x, y = xy 
    j = Int(y)
    if isempty(o.stats)
        o.stats = [FitNormal() for i in 1:size(x,2), j in 1:size(o.stats,2)]
    end
    for i in 1:nvars(o)
        _fit!(o.stats[i, j], x[i])
    end
end


function classify(o::FastNode)
    out = 1
    n = nobs(o.stats[1])
    for j in 2:nkeys(o)
        n2 = nobs(o.stats[1, j])
        if n2 > n 
            out = j 
            n = n2
        end
    end
    out
end