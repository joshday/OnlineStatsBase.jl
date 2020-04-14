module OnlineStatsBase

using Statistics, Dates, LinearAlgebra
using OrderedCollections: OrderedDict

import StatsBase: StatsBase, nobs, fit!
import AbstractTrees: AbstractTrees

export
    OnlineStat,
    # functions
    nobs, value, fit!,
    # Weights
    EqualWeight, ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, McclainWeight,
    # Stats
    Counter, CountMap, CountMissing, CovMatrix, Extrema, FTSeries, Group, GroupBy, Mean, Moments, 
    Part, Series, Sum, Variance

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{T} end
input(o::OnlineStat{T}) where {T} = T
nobs(o::OnlineStat) = o.n

# Stats that hold a collection of other stats
abstract type StatCollection{T} <: OnlineStat{T} end
Base.show(io::IO, o::StatCollection) = AbstractTrees.print_tree(io, o)

AbstractTrees.printnode(io::IO, o::StatCollection) = print(io, name(o, false, false))
AbstractTrees.children(o::StatCollection) = collect(o.stats)

struct SameLine 
    items 
    sep
    SameLine(items, sep='|') = new(items, sep)
end
AbstractTrees.printnode(io::IO, o::SameLine) = print(io, join(o.items, o.sep))


"""
    value(stat::OnlineStat)

Calculate the value of `stat` from its "sufficient statistics".
"""
@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

#-----------------------------------------------------------------------# Base
Base.:(==)(o::OnlineStat, o2::OnlineStat) = false
function Base.:(==)(o1::T, o2::T) where {T<:OnlineStat}
    nms = fieldnames(typeof(o1))
    all(getfield.(Ref(o1), nms) .== getfield.(Ref(o2), nms))
end

Base.copy(o::OnlineStat) = deepcopy(o)

"""
    merge!(stat1, stat2)

Merge `stat1` into `stat2` (supported by most `OnlineStat` types).

# Example

    a = fit!(Mean(), 1:10)
    b = fit!(Mean(), 11:20)
    merge!(a, b)
"""
function Base.merge!(o::OnlineStat, o2::OnlineStat)
    (nobs(o) > 0 || nobs(o2) > 0) && _merge!(o, o2)
    o
end
_merge!(o, o2) = @warn("Merging $(name(o2)) into $(name(o)) is not well-defined.  No merging occurred.")
Base.merge(o::OnlineStat, o2::OnlineStat) = merge!(copy(o), o2)

#-----------------------------------------------------------------------# Show
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o, false, false), ": ")
    print(io, "n=", nobs(o))
    print(io, " | value=")
    show(IOContext(io, :compact => true), value(o))
end
function name(T::Type, withmodule = false, withparams = true)
    s = string(T)
    if !withmodule
        s = replace(s, r"([a-zA-Z]*\.)" => "")  # remove text that ends in period
    end
    if !withparams
        s = replace(s, r"\{(.*)" => "")  # remove "{" to the end of the string
    end
    s
end
name(o, args...) = name(typeof(o), args...)

#-----------------------------------------------------------------------# fit!
"""
    fit!(stat::OnlineStat, data)

Update the "sufficient statistics" of a `stat` with more data.   If `typeof(data)` is not
the type of a single observation for the provided `stat`, `fit!` will attempt to iterate
through and `fit!` each item in `data`.  Therefore, `fit!(Mean(), 1:10)` translates
roughly to:

```
o = Mean()
for x in 1:10
    fit!(o, x)
end
```
"""
fit!(o::OnlineStat{T}, yi::T) where {T} = (_fit!(o, yi); return o)

function fit!(o::OnlineStat{I}, y::T) where {I, T}
    T == eltype(y) && error("The input for $(name(o,false,false)) is a $I.  Found $T.")
    for yi in y
        fit!(o, yi)
    end
    o
end

#-----------------------------------------------------------------------# utils
"""
    smooth(a, b, γ)

Weighted average of `a` and `b` with weight `γ`.
"""
smooth(a, b, γ) = a + γ * (b - a)

"""
    smooth!(a, b, γ)

Update `a` in place by applying the [`smooth`](@ref) function elementwise with `b`.
"""
function smooth!(a, b, γ)
    for (i, bi) in zip(eachindex(a), b)
        a[i] = smooth(a[i], bi, γ)
    end
end

"""
    smooth_syr!(A::AbstractMatrix, x, γ::Number)

Weighted average of symmetric rank-1 update.  Updates the upper triangle of:

`A = (1 - γ) * A + γ * x * x'`
"""
function smooth_syr!(A::AbstractMatrix, x, γ)
    for j in 1:size(A, 2), i in 1:j
        A[i, j] = smooth(A[i,j], x[i] * conj(x[j]), γ)
    end
end

# bessel correction
bessel(o) = nobs(o) / (nobs(o) - 1)

Statistics.std(o::OnlineStat; kw...) = sqrt.(var(o; kw...))

const TwoThings{T,S} = Union{Tuple{T,S}, Pair{<:T,<:S}, NamedTuple{names, Tuple{T,S}}} where names

neighbors(x) = @inbounds ((x[i], x[i+1]) for i in eachindex(x)[1:end-1])

#-----------------------------------------------------------------------# includes
include("weight.jl")
include("stats.jl")
include("part.jl")
end
