#-----------------------------------------------------------------------# FastNode
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
nobs(o::FastNode) = sum(nobs, o.stats[1, :])
probs(o::FastNode) = nobs.(o.stats[1, :]) ./ nobs(o)
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
child(o::FastNode, x::VectorOb) = x[o.j] < o.at ? first(o.children) : last(o.children)

# node, tree_length --> left, right
function split(o::FastNode, d::Int, split_candidates::Vector{Float64})
    n = nobs(o)
    pl = zeros(nkeys(o))  # "prob" left
    pr = zeros(nkeys(o))  # "prob" right
    ent_root = impurity(probs(o))
    ig = -Inf
    ind = 0 
    at = -Inf
    for j in 1:nvars(o)
        stats_j = o.stats[j, :]
        k = 0 
        for stat in stats_j 
            μ = mean(stat)
            σ = std(stat)
            split_candidates[k+1] = μ - 2σ
            split_candidates[k+2] = μ - 1.5σ
            split_candidates[k+3] = μ - σ
            split_candidates[k+4] = μ - .5σ
            split_candidates[k+5] = μ 
            split_candidates[k+6] = μ + .5σ
            split_candidates[k+7] = μ + σ
            split_candidates[k+8] = μ + 1.5σ
            split_candidates[k+9] = μ + 2σ
            k += 9
        end
        for loc in split_candidates
            for k in 1:nkeys(o)
                pl[k] = cdf(stats_j[k], loc)
                pr[k] = 1.0 - pl[k]
            end
            ent_l = impurity(pl ./ sum(pl))
            ent_r = impurity(pr ./ sum(pr))
            ent_after = smooth(ent_l, ent_r, sum(pr) / (sum(pr) + sum(pl)))
            new_ig = ent_root - ent_after 
            if new_ig > ig 
                ig = new_ig 
                ind = j 
                at = loc
            end
        end
    end
    o.j = ind 
    o.at = at
    o.ig = ig
    push!(o.children, d + 1)
    push!(o.children, d + 2)
    FastNode(o; id = d + 1), FastNode(o; id = d + 2)
end
impurity(p) = entropy(p, 2)