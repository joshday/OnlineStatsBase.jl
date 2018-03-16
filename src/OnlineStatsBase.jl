__precompile__(true)
module OnlineStatsBase

import NamedTuples
import LearnBase: value

#-----------------------------------------------------------------------# Data
const VectorOb = Union{AbstractVector, Tuple, NamedTuples.NamedTuple} # 1 
const XyOb     = Tuple{VectorOb, Any}              # (1, 0)

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{N} end

#-----------------------------------------------------------------------# fit! and value
@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end
_fit!(o::OnlineStat, args...) = error("$o hasn't implemented `_fit!` yet.")

#-----------------------------------------------------------------------# show
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o, false, false), ": ")
    show(IOContext(io, :compact => true), value(o))
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
function Base.merge!(o::OnlineStat, o2::OnlineStat)
    warn("Merging $(name(o2)) into $(name(o)) is not well-defined.  No merging occurred.")
end
Base.merge(o::OnlineStat, o2::OnlineStat, γ) = merge!(copy(o), o2, γ)


#-----------------------------------------------------------------------# Weight
# abstract type Weight end 
# include("weight.jl")

#-----------------------------------------------------------------------# name
# Example:
# name(o::OnlineStats.CountMap{Int}, false, true)   --> CountMap{Int}
# name(o::OnlineStats.CountMap{Int}, false, false)  --> CountMap
# name(o::OnlineStats.CountMap{Int}, true, true)    --> OnlineStats.CountMap{Int}
# name(o::OnlineStats.CountMap{Int}, true, false)   --> OnlineStats.CountMap
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

end
