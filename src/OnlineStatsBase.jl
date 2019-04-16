module OnlineStatsBase

using Statistics, Dates, OrderedCollections

import LearnBase: nobs, value, fit!
import StatsBase: StatsBase

export
    OnlineStat, Weight,
    nobs, value, fit!, eachrow, eachcol,
    EqualWeight, ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, McclainWeight,
    # Stats
    Counter, CountMap, Extrema, FTSeries, Group, GroupBy, Mean, Moments, Series, Sum, Variance

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{T} end

# Stats that hold a collection of other stats
abstract type StatCollection{T} <: OnlineStat{T} end
function Base.show(io::IO, o::StatCollection)
    print(io, name(o, false, false))
    print_stat_tree(io, o.stats)
end
function print_stat_tree(io::IO, stats)
    for (i, stat) in enumerate(stats)
        char = i == length(stats) ? '└' : '├'
        print(io, "\n  $(char)── $stat")
    end
end

nobs(o::OnlineStat) = o.n

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
function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        s = replace(s, r"([a-zA-Z]*\.)" => "")  # remove text that ends in period
    end
    if !withparams
        s = replace(s, r"\{(.*)" => "")  # remove "{" to the end of the string
    end
    s
end

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
function _fit!(o::OnlineStat{T}, arg) where {T}
    error("A $(typeof(arg)) is not a single observation for $((name(o, false, true)))")
end

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
function smooth_syr!(A::AbstractMatrix, x, γ::Number)
    for j in 1:size(A, 2), i in 1:j
        A[i, j] = smooth(A[i,j], x[i] * conj(x[j]), γ)
    end
end

# bessel correction
bessel(o) = nobs(o) / (nobs(o) - 1)

Statistics.std(o::OnlineStat; kw...) = sqrt.(var(o; kw...))

input(o::OnlineStat{T}) where {T} = T

const TwoThings{T,S} = Union{Tuple{T,S}, Pair{T,S}, NamedTuple{names, Tuple{T,S}}} where names

#-----------------------------------------------------------------------# Compat
@static if VERSION < v"1.1.0"
    export eachrow

    """
        eachrow(x::AbstractVecOrMat)

    Iterator over the rows of `x` as views.

    # Example
        for xi in eachrow(rand(3,2))
            println(xi)
        end
    """
    eachrow(A::Union{AbstractVector, AbstractMatrix}) = (view(A, i, :) for i in axes(A, 1))
else
    import Base: eachrow
end

@deprecate eachrow(x::AbstractMatrix, y::AbstractVector) zip(eachrow(x), y)

# #-----------------------------------------------------------------------# OnlineIterator
# struct OnlineIterator{R,T,S}
#     thing::T
#     buffer::S
#     OnlineIterator{R}(thing::T, buffer::S) where {T,S,R} = new{R,T,S}(thing, buffer)
# end

# Base.iterate(o::OnlineIterator, i=1) = i > length(o) ? nothing : (o[i], i+1)
# Base.keys(o::OnlineIterator) = Base.OneTo(length(o))


# eachrow(args...) = eachrow(args)

# """
#     eachcol(x::AbstractMatrix)
#     eachcol(x::AbstractMatrix, y::AbstractVector)

# Iterator over the columns of `x` (paired in a tuple with the values of `y`).

# # Example
#     for xi in eachcol(rand(3,2))
#         println(xi)
#     end
# """
# eachcol(args...) = eachcol(args)

# # helpers
# function copyrow!(buffer::Vector, x::AbstractMatrix, i::Int)
#     for j in eachindex(buffer)
#         buffer[j] = x[i, j]
#     end
#     buffer
# end
# function copycol!(buffer::Vector, x::AbstractMatrix, j::Int)
#     for i in eachindex(buffer)
#         buffer[i] = x[i, j]
#     end
#     buffer
# end

# # Matrix rows
# Base.length(o::OnlineIterator{:row, <:AbstractMatrix}) = size(o.thing, 1)
# Base.getindex(o::OnlineIterator{:row, <:AbstractMatrix}, i::Int) = copyrow!(o.buffer, o.thing, i)
# eachrow(m::AbstractMatrix{T}) where {T} = OnlineIterator{:row}(m, Vector{T}(undef, size(m, 2)))

# # Matrix cols
# Base.length(o::OnlineIterator{:col, <:AbstractMatrix}) = size(o.thing, 2)
# Base.getindex(o::OnlineIterator{:col, <:AbstractMatrix}, i::Int) = copycol!(o.buffer, o.thing, i)
# eachcol(m::AbstractMatrix{T}) where {T} = OnlineIterator{:col}(m, Vector{T}(undef, size(m, 1)))


# const XY = Tuple{T, S} where {T<:AbstractMatrix, S<:AbstractVector}

# # XY rows
# Base.length(o::OnlineIterator{:row, <:XY}) = size(o.thing[1], 1)
# Base.getindex(o::OnlineIterator{:row, <:XY}, i::Int) = (copyrow!(o.buffer, o.thing[1], i), o.thing[2][i])
# eachrow(t::XY) = OnlineIterator{:row}(t, Vector{eltype(t[1])}(undef, size(t[1], 2)))

# # XY cols
# Base.length(o::OnlineIterator{:col, <:XY}) = size(o.thing[1], 2)
# Base.getindex(o::OnlineIterator{:col, <:XY}, i::Int) = (copycol!(o.buffer, o.thing[1], i), o.thing[2][i])
# eachcol(t::XY) = OnlineIterator{:col}(t, Vector{eltype(t[1])}(undef, size(t[1], 1)))

include("weight.jl")
include("stats.jl")
end
