__precompile__(true)
module OnlineStatsBase

#-----------------------------------------------------------------------# Data
const ScalarOb = Union{Number, AbstractString, Symbol}  # 0
const VectorOb = Union{AbstractVector, Tuple}           # 1 
const XyOb     = Tuple{VectorOb, ScalarOb}              # (1, 0)
const Data = Union{ScalarOb, VectorOb, AbstractMatrix, XyOb}

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{N} end

"An OnlineStat which can be updated exactly."
abstract type ExactStat{N}      <: OnlineStat{N} end

"An OnlineStat which must be updated approximately."
abstract type StochasticStat{N} <: OnlineStat{N} end

# The default value(o) returns the first field
"""
    value(o::OnlineStat)

Return the value of the OnlineStat.
"""
@generated function _value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end
_fit!(o::OnlineStat, ob, γ::Float64) = error("typeof(o) needs method: OnlineStatsBase._fit!")

function Base.show(io::IO, o::OnlineStat)
    print(io, name(o), "(")
    showcompact(io, _value(o))
    print(io, ")")
end

function Base.:(==)(o1::OnlineStat, o2::OnlineStat)
    typeof(o1) == typeof(o2) || return false
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end

Base.copy(o::OnlineStat) = deepcopy(o)

function Base.merge!(o::T, o2::T, γ::Float64) where {T<:OnlineStat} 
    warn("Merging not well-defined for $(typeof(o)).  No merging occurred.")
end
Base.merge(o::T, o2::T, γ::Float64) where {T<:OnlineStat} = merge!(copy(o), o2, γ)

default_weight(o::OnlineStat)       = error("$(typeof(o)) has no `default_weight` method")
default_weight(o::ExactStat)        = EqualWeight()
default_weight(o::StochasticStat)   = LearningRate()

function default_weight(t::Tuple)
    W = default_weight(first(t))
    all(default_weight.(t) .== W) ||
        error("Weight must be specified when defaults differ.  Found: $(name.(default_weight.(t))).")
    return W
end

#-----------------------------------------------------------------------# Weight
"""
`Weight` is an abstract type.  Subtypes must be callable have a method to produce the
weight given the current number of observations in an OnlineStat `n` and the number of 
observations included in the update (`n2`).

```
MyWeight(n, n2 = 1)
```
"""
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
