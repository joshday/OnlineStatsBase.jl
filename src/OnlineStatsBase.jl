__precompile__(true)
module OnlineStatsBase

import LearnBase: fit!, nobs, value, predict
import StatsBase: autocov, autocor, confint
import DataStructures: OrderedDict

export 
# functions 
    fit!, nobs, value, autocov, autocor, predict, confint, probs,
# weights 
    EqualWeight, ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, 
    McclainWeight, Bounded, Scaled,
# stats
    AutoCov,
    Bootstrap,
    Count,
    CountMap,
    CovMatrix,
    CStat,
    Diff,
    Extrema,
    FitBeta, FitCauchy, FitGamma, FitLogNormal, FitNormal, FitMultinomial, FitMVNormal,
    Group,
    HyperLogLog,
    Lag,
    Mean,
    Moments,
    ProbMap,
    ReservoirSample,
    Series, FTSeries,
    Sum,
    Variance



const Tup      = Union{Tuple, NamedTuple}
const VectorOb = Union{AbstractVector, Tup} # 1 
const XyOb     = Tuple{VectorOb, Any}              # (1, 0)

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{N} end

nobs(o::OnlineStat) = o.n

@generated function value(o::OnlineStat)
    r = first(fieldnames(o))
    return :(o.$r)
end

#-----------------------------------------------------------------------# utils 
smooth(a, b, γ) = a + γ * (b - a)
function smooth!(a, b, γ)
    for i in eachindex(a)
        a[i] = smooth(a[i], b[i], γ)
    end
end
function smooth_syr!(A::AbstractMatrix, x, γ::Number)
    for j in 1:size(A, 2), i in 1:j
        A[i, j] = smooth(A[i,j], x[i] * x[j], γ)
    end
end

unbias(o) = nobs(o) / (nobs(o) - 1)
Base.std(o::OnlineStat; kw...) = sqrt.(var(o; kw...))

#-----------------------------------------------------------------------# fit!
_fit!(o::OnlineStat, arg) = error("$o hasn't implemented `_fit!` yet.")

fit!(o::OnlineStat{0}, y) = (_fit!(o, y); o)
function fit!(o::OnlineStat{0}, y::Union{VectorOb, AbstractArray})
    for yi in y 
        fit!(o, yi)
    end 
    o
end

fit!(o::OnlineStat{1}, y::VectorOb) = (_fit!(o, y); o)
function fit!(o::OnlineStat{1}, y::AbstractMatrix, dim::Int = 1)
    n, p = size(y)
    buffer = Vector{eltype(y)}(undef, p)
    if dim == 1
        for i in 1:n
            for j in 1:p
                @inbounds buffer[j] = y[i, j]
            end
            fit!(o, buffer)
        end
    elseif dim == 2 
        for i in 1:p
            for j in 1:n
                @inbounds buffer[j] = y[j, i]
            end
            fit!(o, buffer)
        end
    else 
        error("dim must be 1 or 2.")
    end
    o
end

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
    @warn("Merging $(name(o2)) into $(name(o)) is not well-defined.  No merging occurred.")
end

Base.merge(o::OnlineStat, o2::OnlineStat, γ) = merge!(copy(o), o2, γ)



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

#-----------------------------------------------------------------------# includes 
include("weight.jl")
include("stats.jl")
end
