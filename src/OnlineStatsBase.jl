__precompile__(true)
module OnlineStatsBase

using Compat
import LearnBase: nobs, value

export EqualWeight, ExponentialWeight, LearningRate, LearningRate2, McclainWeight, 
    HarmonicWeight

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{N} end

nobs(o::OnlineStat) = o.n

@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

_fit!(o::OnlineStat, arg) = error("$o hasn't implemented `_fit!(stat, observation)` yet.")

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
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o, false, false), ": ")
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

#-----------------------------------------------------------------------# Weight
include("weight.jl")
end
