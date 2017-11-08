__precompile__(true)
module OnlineStatsBase

import LearnBase: value, fit!, nobs

#-----------------------------------------------------------------------# Data
const ScalarOb = Union{Number, AbstractString, Symbol}  # 0
const VectorOb = Union{AbstractVector, Tuple}           # 1 
const XyOb     = Tuple{VectorOb, ScalarOb}              # (1, 0)
const Data = Union{ScalarOb, VectorOb, AbstractMatrix, XyOb}

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{I} end

"An OnlineStat which can be updated exactly."
abstract type ExactStat{N}      <: OnlineStat{N} end

"An OnlineStat which must be updated approximately."
abstract type StochasticStat{N} <: OnlineStat{N} end

# The default value(o) returns the first field
@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

# Base functions
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o), "(")
    showcompact(io, value(o))
    print(io, ")")
end

function Base.:(==)(o1::OnlineStat, o2::OnlineStat)
    typeof(o1) == typeof(o2) || return false
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end

function Base.merge!(o::T, o2::T, γ::Float64) where {T<:OnlineStat} 
    warn("Merging not well-defined for $(typeof(o)).  No merging occurred.")
end
Base.merge(o::T, o2::T, γ::Float64) where {T<:OnlineStat} = merge!(copy(o), o2, γ)

default_weight(o::OnlineStat) = error("$(typeof(o)) needs to overload `default_weight`")
default_weight(o::ExactStat) = EqualWeight()
default_weight(o::StochasticStat) = LearningRate()
function default_weight(t::Tuple)
    W = default_weight(first(t))
    all(default_weight.(t) .== W) ||
        error("Weight must be specified when defaults differ.  Found: $(name.(default_weight.(t))).")
    return W
end

#-----------------------------------------------------------------------# Weight
abstract type Weight end 
include("weight.jl")

#-----------------------------------------------------------------------# helpers
function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        # remove text that ends in period:  OnlineStats.Mean -> Mean
        s = replace(s, r"([a-zA-Z]*\.)", "")
    end
    if !withparams
        # replace everything from "{" to the end of the string
        s = replace(s, r"\{(.*)", "")
    end
    s
end



end #module
