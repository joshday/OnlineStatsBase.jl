module OnlineStatsBase

import LearnBase: nobs, value, fit!
export 
    # abstract types
    OnlineStat, Weight,
    # functions
    nobs, value, fit!, _fit!, eachrow, eachcol,
    # Weights
    EqualWeight, ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, McclainWeight,
    # OnlineIterator
    OnlineIterator

abstract type OnlineStat{T} end

nobs(o::OnlineStat) = o.n

"""
    value(o::OnlineStat)

Calculate the value of the stat from its "sufficient statistics".
"""
@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

_fit!(o::OnlineStat{T}, arg) where {T} = 
    error("$(name(o, false, true)) doesn't know how to fit an item of type $(typeof(arg))")

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
    fit!(o::OnlineStat, data)

Update the "sufficient statistics" of a stat with more data.
"""
function fit!(o::OnlineStat{T}, yi::T) where {T}
    _fit!(o, yi)
    o
end

function fit!(o::OnlineStat{I}, y::T) where {I, T}
    T == eltype(y) && error("The input for $(name(o,false,false)) is a $I.  Found $T.")
    for yi in y 
        fit!(o, yi)
    end
    o
end

fit!(o::OnlineStat, y::Nothing) = nothing

#-----------------------------------------------------------------------# OnlineIterator
struct OnlineIterator{R,T,S}
    thing::T 
    buffer::S
    OnlineIterator{R}(thing::T, buffer::S) where {T,S,R} = new{R,T,S}(thing, buffer)
end

Base.iterate(o::OnlineIterator, i=1) = i > length(o) ? nothing : (o[i], i+1)
Base.keys(o::OnlineIterator) = Base.OneTo(length(o))
eachrow(args...) = eachrow(args)
eachcol(args...) = eachcol(args)

# helpers 
function copyrow!(buffer::Vector, x::AbstractMatrix, i::Int)
    for j in eachindex(buffer)
        buffer[j] = x[i, j]
    end
    buffer
end
function copycol!(buffer::Vector, x::AbstractMatrix, j::Int)
    for i in eachindex(buffer)
        buffer[i] = x[i, j]
    end
    buffer
end

# Matrix rows
Base.length(o::OnlineIterator{:row, <:AbstractMatrix}) = size(o.thing, 1)
Base.getindex(o::OnlineIterator{:row, <:AbstractMatrix}, i::Int) = copyrow!(o.buffer, o.thing, i)
eachrow(m::AbstractMatrix{T}) where {T} = OnlineIterator{:row}(m, Vector{T}(undef, size(m, 2)))

# Matrix cols 
Base.length(o::OnlineIterator{:col, <:AbstractMatrix}) = size(o.thing, 2)
Base.getindex(o::OnlineIterator{:col, <:AbstractMatrix}, i::Int) = copycol!(o.buffer, o.thing, i)
eachcol(m::AbstractMatrix{T}) where {T} = OnlineIterator{:col}(m, Vector{T}(undef, size(m, 1)))

# XY rows 
const XY = Tuple{T, S} where {T<:AbstractMatrix, S<:AbstractVector}

Base.length(o::OnlineIterator{:row, <:XY}) = size(o.thing[1], 1)
Base.getindex(o::OnlineIterator{:row, <:XY}, i::Int) = (copyrow!(o.buffer, o.thing[1], i), o.thing[2][i])
eachrow(t::XY) = OnlineIterator{:row}(t, Vector{eltype(t[1])}(undef, size(t[1], 2)))

# XY cols 
Base.length(o::OnlineIterator{:col, <:XY}) = size(o.thing[1], 2)
Base.getindex(o::OnlineIterator{:col, <:XY}, i::Int) = (copycol!(o.buffer, o.thing[1], i), o.thing[2][i])
eachcol(t::XY) = OnlineIterator{:col}(t, Vector{eltype(t[1])}(undef, size(t[1], 1)))

#-----------------------------------------------------------------------# Weight
include("weight.jl")
end
