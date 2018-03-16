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

function _predict(o::NBClassifier, x::VectorOb, p = zeros(nkeys(o)), n = nobs(o))
    for (k, gk) in enumerate(values(o))
        # P(Ck)
        p[k] = log(nobs(gk) / n + ϵ) 
        # P(xj | Ck)
        for j in 1:length(x)
            p[k] += log(pdf(gk[j], x[j]) + ϵ)
        end
        p[k] = exp(p[k])
    end
    sp = sum(p)
    sp == 0.0 ? p : scale!(p, inv(sp))
end
function _classify(o::NBClassifier, x::VectorOb, p = zeros(nkeys(o)), n = nobs(o)) 
    _, k = findmax(_predict(o, x, p, n))
    index_to_key(o, k)
end
function index_to_key(d, i)
    for (k, ky) in enumerate(keys(d))
        k == i && return ky 
    end
end
predict(o::NBClassifier, x::VectorOb) = _predict(o, x)
classify(o::NBClassifier, x::VectorOb) = _classify(o, x)
function predict(o::NBClassifier, x::AbstractMatrix)
    n = nobs(o)
    p = zeros(nkeys(o))
    mapslices(xi -> _predict(o, xi, p, n), x, 2)
end
function classify(o::NBClassifier, x::AbstractMatrix)
    n = nobs(o)
    p = zeros(nkeys(o))
    mapslices(xi -> _classify(o, xi, p, n), x, 2)
end
function classify_node(o::NBClassifier)
    _, k = findmax([nobs(g) for g in values(o)])
    index_to_key(o, k)
end