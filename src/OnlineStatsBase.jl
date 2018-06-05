__precompile__(true)
module OnlineStatsBase

using Compat
import LearnBase: nobs, value, fit!
export nobs, value, fit!, _fit!, eachrow, eachcol, Weight, OnlineStat, EqualWeight, 
    ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, McclainWeight

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
    error("$(name(o, false, true)) doesn't know how to fit an item of type $(typeof(arg)) ")

#-----------------------------------------------------------------------# Base 
Base.:(==)(o::OnlineStat, o2::OnlineStat) = false 
function Base.:(==)(o1::T, o2::T) where {T<:OnlineStat}
    nms = fieldnames(typeof(o1))
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end

Base.copy(o::OnlineStat) = deepcopy(o)
function Base.merge!(o::OnlineStat, o2::OnlineStat)
    Compat.@warn("Merging $(name(o2)) into $(name(o)) is not well-defined.  No merging occurred.")
end
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

fit!(o::OnlineStat, y::Compat.Nothing) = nothing

#-----------------------------------------------------------------------# Weight
include("weight.jl")
end
