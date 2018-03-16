#-----------------------------------------------------------------------# NBClassifier
mutable struct NBClassifier{T, G<:Group, F} <: OnlineStat{(1, 0)}
    d::OrderedDict{T, G}
    init::F
    # For trees
    id::Int 
    j::Int 
    at::Union{Number, String, Symbol, Char}
    ig::Float64
end
function NBClassifier(T::Type, init::Function) 
    G = typeof(init())
    NBClassifier(OrderedDict{T, G}(), init, 1, 0, -Inf, -Inf)
end
function Base.show(io::IO, o::NBClassifier)
    print(io, "NBClassifier")
    for (k, p) in zip(keys(o), probs(o))
        print(io, "\n    > $k ($(round(p, 4)))")
    end
end
function _fit!(o::NBClassifier, xy)
    x, y = xy 
    if haskey(o.d, y)
        fit!(o.d[y], x)
    else 
        stat = o.init()
        fit!(stat, x)
        o.d[y] = stat
    end
end

Base.keys(o::NBClassifier) = keys(o.d)
Base.values(o::NBClassifier) = values(o.d)
nkeys(o::NBClassifier) = length(o.d)
nvars(o::NBClassifier) = length(o.init)
nobs(o::NBClassifier) = isempty(o.d) ? 0 : sum(nobs, values(o))
probs(o::NBClassifier) = isempty(o.d) ? zeros(0) : map(nobs, values(o)) ./ nobs(o)

