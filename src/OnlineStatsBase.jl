__precompile__(true)
module OnlineStatsBase

import LearnBase: value, ObsDim, ObsDimension
import StatsBase: Histogram, skewness, kurtosis, confint, coef, predict, nobs, fit!

export
    # Series
    Series,
    # Weight
    Weight,
    EqualWeight, BoundedEqualWeight, ExponentialWeight, LearningRate, LearningRate2, McclainWeight,
    HarmonicWeight, Bounded, Scaled,
    # OnlineStats
    OnlineStat,
    CovMatrix, Diff, Extrema, HyperLogLog, KMeans, Mean, Moments, MV,OHistogram, OrderStats,
    QuantileMM, ReservoirSample, RidgeReg, Sum, Variance,
    # Other
    Bootstrap,
    # functions
    nobs, fit!, value, stats, Rows, Cols, predict, coef, replicates

#-----------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{I, W} end

# Base functions
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o), "(")
    showcompact(io, value(o))
    print(io, ")")
end
Base.copy(o::OnlineStat) = deepcopy(o)
function Base.:(==){T <: OnlineStat}(o1::T, o2::T)
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end
Base.merge{T <: OnlineStat}(o::T, o2::T, wt::Float64) = merge!(copy(o), o2, wt)

# OnlineStat Interface (sans `fit!`)
value(o::OnlineStat) = getfield(o, fieldnames(o)[1])
input_ndims{I}(o::OnlineStat{I}) = I
default_weight{I, W}(o::OnlineStat{I, W}) = W()

function input_ndims(t::Tuple)
    I = input_ndims(first(t))
    for ti in t
        input_ndims(ti) == I || error("Inputs don't match. Found: $(input_ndims.(t))")
    end
    # all(input_ndims.(t) .== I) ||
    #     error("Inputs don't match. Found: $(input_ndims.(t))")
    return I
end

function default_weight(t::Tuple)
    W = default_weight(first(t))
    all(default_weight.(t) .== W) ||
        error("Default weights don't match.  Found: $(default_weight.(t))")
    return W
end


#-----------------------------------------------------------------------# Show helpers
function show_fields(io::IO, o)
    nms = fields_to_show(o)
    print(io, "(")
    for nm in nms
        print(io, "$nm = $(getfield(o, nm))")
        nm != nms[end] && print(io, ", ")
    end
    print(io, ")")
end

fields_to_show(o) = fieldnames(o)

header(io::IO, s::AbstractString) = println(io, "▦ $s" )

function name(o, withmodule = false)
    s = string(typeof(o))
    if !withmodule
        # remove text that ends in period:  OnlineStats.Mean -> Mean
        s = replace(s, r"([a-zA-Z]*\.)", "")
    end
    # if !withparams
    #     # replace everything from "{" to the end of the string
    #     s = replace(s, r"\{(.*)", "")
    # end
    s
end

#-----------------------------------------------------------------------# Common
smooth(x, y, γ) = x + γ * (y - x)

function smooth!(x, y, γ)
    length(x) == length(y) || throw(DimensionMismatch("can't smooth arrays of different length"))
    for i in eachindex(x)
        @inbounds x[i] = smooth(x[i], y[i], γ)
    end
end

function smooth_syr!(A::AbstractMatrix, x, γ::Float64)
    size(A, 1) == length(x) || throw(DimensionMismatch())
    for j in 1:size(A, 2), i in 1:j
        @inbounds A[i, j] = (1.0 - γ) * A[i, j] + γ * x[i] * x[j]
    end
end

unbias(o) = o.nobs / (o.nobs - 1)

const ϵ = 1e-6

fit!(o::OnlineStat{(1,0)}, t::Tuple, γ) = fit!(o, t..., γ)


#-----------------------------------------------------------------------# includes
include("weight.jl")
include("series.jl")
include("stats.jl")
include("mv.jl")
include("bootstrap.jl")
end
