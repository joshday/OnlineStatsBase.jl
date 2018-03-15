__precompile__(true)
module OnlineStatsBase

import NamedTuples
import LearnBase: fit!, value

#-----------------------------------------------------------------------# Data
const VectorOb = Union{AbstractVector, Tuple, NamedTuples.NamedTuple} # 1 
const XyOb     = Tuple{VectorOb, Any}              # (1, 0)

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{N} end
default_weight(o::OnlineStat) = error("$(typeof(o)) needs a `default_weight` method.")

abstract type ExactStat{N} <: OnlineStat{N} end
default_weight(o::ExactStat) = EqualWeight()

abstract type StochasticStat{N} <: OnlineStat{N} end
default_weight(o::StochasticStat) = LearningRate()

function default_weight(t::Union{Tuple, NamedTuples.NamedTuple})
    W = default_weight(first(t))
    for item in t 
        default_weight(item) != W && 
        error("Weight must be specified when defaults differ. Found:")
    end
    return W
end

#-----------------------------------------------------------------------# fit! and value
@deprecate _value(o::OnlineStat) value(o::OnlineStat)
@deprecate _fit!(o::OnlineStat, y, w) fit!(o::OnlineStat, y, w)

@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

#-----------------------------------------------------------------------# show
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o), "(")
    showcompact(io, value(o))
    print(io, ")")
end

#-----------------------------------------------------------------------# ==
function Base.:(==)(o1::OnlineStat, o2::OnlineStat)
    typeof(o1) == typeof(o2) || return false
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end

#-----------------------------------------------------------------------# copy
Base.copy(o::OnlineStat) = deepcopy(o)

#-----------------------------------------------------------------------# merge
function Base.merge!(o::OnlineStat, o2::OnlineStat, γ)
    warn("Merging $(name(o2)) into $(name(o)) is not well-defined.  No merging occurred.")
end
Base.merge(o::OnlineStat, o2::OnlineStat, γ) = merge!(copy(o), o2, γ)


#-----------------------------------------------------------------------# Weight
abstract type Weight end 
include("weight.jl")

#-----------------------------------------------------------------------# name
# Example:
# name(o::OnlineStats.CountMap{Int}, false, true)   --> CountMap{Int}
# name(o::OnlineStats.CountMap{Int}, false, false)  --> CountMap
# name(o::OnlineStats.CountMap{Int}, true, true)    --> OnlineStats.CountMap{Int}
# name(o::OnlineStats.CountMap{Int}, true, false)   --> OnlineStats.CountMap
function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        s = replace(s, r"([a-zA-Z]*\.)", "")  # remove text that ends in period
    end
    if !withparams
        s = replace(s, r"\{(.*)", "")  # remove "{" to the end of the string
    end
    s
end

end
