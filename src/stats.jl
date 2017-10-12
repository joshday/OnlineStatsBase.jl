#-----------------------------------------------------------------------# CovMatrix
"""
    CovMatrix(d)
Covariance Matrix of `d` variables.

    y = randn(100, 5)
    Series(y, CovMatrix(5))
"""
mutable struct CovMatrix <: OnlineStat{1, EqualWeight}
    value::Matrix{Float64}
    cormat::Matrix{Float64}
    A::Matrix{Float64}  # X'X / n
    b::Vector{Float64}  # X * 1' / n (column means)
    nobs::Int
    CovMatrix(p::Integer) = new(zeros(p, p), zeros(p, p), zeros(p, p), zeros(p), 0)
end
function fit!(o::CovMatrix, x::VectorOb, γ::Float64)
    smooth!(o.b, x, γ)
    smooth_syr!(o.A, x, γ)
    o.nobs += 1
    o
end
function value(o::CovMatrix)
    o.value[:] = full(Symmetric((o.A - o.b * o.b')))
    scale!(o.value, unbias(o))
end
Base.length(o::CovMatrix) = length(o.b)
Base.mean(o::CovMatrix) = o.b
Base.cov(o::CovMatrix) = value(o)
Base.var(o::CovMatrix) = diag(value(o))
Base.std(o::CovMatrix) = sqrt.(var(o))
function Base.cor(o::CovMatrix)
    copy!(o.cormat, value(o))
    v = 1.0 ./ sqrt.(diag(o.cormat))
    scale!(o.cormat, v)
    scale!(v, o.cormat)
    o.cormat
end
function Base.merge!(o::CovMatrix, o2::CovMatrix, γ::Float64)
    smooth!(o.A, o2.A, γ)
    smooth!(o.b, o2.b, γ)
    o.nobs += o2.nobs
    o
end

#-----------------------------------------------------------------------# Diff
"""
    Diff()
Track the difference and the last value.

    s = Series(randn(1000), Diff())
    value(s)
"""
mutable struct Diff{T <: Real} <: OnlineStat{0, EqualWeight}
    diff::T
    lastval::T
end
Diff() = Diff(0.0, 0.0)
Diff{T<:Real}(::Type{T}) = Diff(zero(T), zero(T))
Base.last(o::Diff) = o.lastval
Base.diff(o::Diff) = o.diff
function fit!{T<:AbstractFloat}(o::Diff{T}, x::Real, γ::Float64)
    v = convert(T, x)
    o.diff = v - last(o)
    o.lastval = v
end
function fit!{T<:Integer}(o::Diff{T}, x::Real, γ::Float64)
    v = round(T, x)
    o.diff = v - last(o)
    o.lastval = v
end

#-----------------------------------------------------------------------# Extrema
"""
    Extrema()
Maximum and minimum.

    s = Series(randn(100), Extrema())
    value(s)
"""
mutable struct Extrema <: OnlineStat{0, EqualWeight}
    min::Float64
    max::Float64
    Extrema() = new(Inf, -Inf)
end
function fit!(o::Extrema, y::Real, γ::Float64)
    o.min = min(o.min, y)
    o.max = max(o.max, y)
    o
end
function Base.merge!(o::Extrema, o2::Extrema, γ::Float64)
    o.min = min(o.min, o2.min)
    o.max = max(o.max, o2.max)
    o
end
value(o::Extrema) = (o.min, o.max)
Base.extrema(o::Extrema) = value(o)

#-----------------------------------------------------------------------# HyperLogLog
# Mostly copy/pasted from StreamStats.jl
hash32(d::Any) = hash(d) % UInt32
maskadd32(x::UInt32, mask::UInt32, add::UInt32) = (x & mask) + add
ρ(s::UInt32) = UInt32(leading_zeros(s)) + 0x00000001
const toInt = Int
const toUInt = UInt

function α(m::UInt32)
    if m == 0x00000010
        return 0.673
    elseif m == 0x00000020
        return 0.697
    elseif m == 0x00000040
        return 0.709
    else # if m >= UInt32(128)
        return 0.7213 / (1 + 1.079 / m)
    end
end

"""
    HyperLogLog(b)  # 4 ≤ b ≤ 16
Approximate count of distinct elements.

    s = Series(rand(1:10, 1000), HyperLogLog(12))
    value(s)
"""
mutable struct HyperLogLog <: OnlineStat{0, EqualWeight}
    m::UInt32
    M::Vector{UInt32}
    mask::UInt32
    altmask::UInt32
    function HyperLogLog(b::Integer)
        !(4 ≤ b ≤ 16) && throw(ArgumentError("b must be an Integer between 4 and 16"))
        m = 0x00000001 << b
        M = zeros(UInt32, m)
        mask = 0x00000000
        for i in 1:(b - 1)
            mask |= 0x00000001
            mask <<= 1
        end
        mask |= 0x00000001
        altmask = ~mask
        new(m, M, mask, altmask)
    end
end
function Base.show(io::IO, counter::HyperLogLog)
    print(io, "HyperLogLog($(counter.m) registers, estimate = $(value(counter)))")
end

function fit!(o::HyperLogLog, v::Any, γ::Float64)
    x = hash32(v)
    j = maskadd32(x, o.mask, 0x00000001)
    w = x & o.altmask
    o.M[j] = max(o.M[j], ρ(w))
    o
end

function value(o::HyperLogLog)
    S = 0.0
    for j in 1:o.m
        S += 1 / (2 ^ o.M[j])
    end
    Z = 1 / S
    E = α(o.m) * toUInt(o.m) ^ 2 * Z
    if E <= 5//2 * o.m
        V = 0
        for j in 1:o.m
            V += toInt(o.M[j] == 0x00000000)
        end
        if V != 0
            E_star = o.m * log(o.m / V)
        else
            E_star = E
        end
    elseif E <= 1//30 * 2 ^ 32
        E_star = E
    else
        E_star = -2 ^ 32 * log(1 - E / (2 ^ 32))
    end
    return E_star
end


#-----------------------------------------------------------------------# KMeans
"""
    KMeans(p, k)
Approximate K-Means clustering of `k` clusters and `p` variables.

    using OnlineStats, Distributions
    d = MixtureModel([Normal(0), Normal(5)])
    y = rand(d, 100_000, 1)
    s = Series(y, LearningRate(.6), KMeans(1, 2))
"""
mutable struct KMeans <: OnlineStat{1, LearningRate}
    value::Matrix{Float64}
    v::Vector{Float64}
    KMeans(p::Integer, k::Integer) = new(randn(p, k), zeros(k))
end
Base.show(io::IO, o::KMeans) = print(io, "KMeans($(value(o)'))")
function fit!(o::KMeans, x::VectorOb, γ::Float64)
    d, k = size(o.value)
    length(x) == d || throw(DimensionMismatch())
    for j in 1:k
        o.v[j] = sum(abs2, x - view(o.value, :, j))
    end
    kstar = indmin(o.v)
    for i in 1:d
        o.value[i, kstar] = smooth(o.value[i, kstar], x[i], γ)
    end
end

#-----------------------------------------------------------------------# LinReg
"""
    LinReg(p, λ::Float64 = 0.0)  # use λ for all parameters
    LinReg(p, λfactor::Vector{Float64})
Ridge regression of `p` variables with elementwise regularization.

    x = randn(100, 10)
    y = x * linspace(-1, 1, 10) + randn(100)
    o = LinReg(10)
    Series((x,y), o)
    value(o)
"""
mutable struct LinReg <: OnlineStat{(1,0), EqualWeight}
    β::Vector{Float64}
    A::Matrix{Float64}
    λfactor::Vector{Float64}
    nobs::Int
    function LinReg(p::Integer, λfactor::Vector{Float64} = zeros(p))
        d = p + 1
        new(zeros(p), zeros(d, d), λfactor, 0)
    end
    LinReg(p::Integer, λ::Float64) = LinReg(p, fill(λ, p))
end
Base.show(io::IO, o::LinReg) = print(io, "LinReg: β($(mean(o.λfactor))) = $(value(o)')")
nobs(o::LinReg) = o.nobs

function matviews(o::LinReg)
    p = length(o.β)
    @views o.A[1:p, 1:p], o.A[1:p, end]
end

function fit!(o::LinReg, x::VectorOb, y::Real, γ::Float64)
    xtx, xty = matviews(o)
    smooth_syr!(xtx, x, γ)
    smooth!(xty, x .* y, γ)
    o.A[end] = smooth(o.A[end], y * y, γ)
    o.nobs += 1
end

function value(o::LinReg)
    xtx, xty = matviews(o)
    A = Symmetric(xtx + Diagonal(o.λfactor))
    if isposdef(A)
        o.β[:] = A \ xty
    end
    return o.β
end

coef(o::LinReg) = value(o)
predict(o::LinReg, x::AbstractVector) = x'coef(o)
predict(o::LinReg, x::AbstractMatrix, dim::Rows = Rows()) = x * coef(o)
predict(o::LinReg, x::AbstractMatrix, dim::Cols) = x'coef(o)

# mse(o::LinReg) = (coef(o); o.S[end] * nobs(o) / (nobs(o) - length(o.β)))
# function coeftable(o::LinReg)
#     β = coef(o)
#     p = length(β)
#     se = stderr(o)
#     ts = β ./ se
#     CoefTable(
#         [β se ts Ds.ccdf(Ds.FDist(1, nobs(o) - p), abs2.(ts))],
#         ["Estimate", "Std.Error", "t value", "Pr(>|t|)"],
#         ["x$i" for i in 1:p],
#         4
#     )
# end
# function confint(o::LinReg, level::Real = 0.95)
#     β = coef(o)
#     mult = stderr(o) * quantile(Ds.TDist(nobs(o) - length(β) - 1), (1 - level) / 2)
#     hcat(β, β) + mult * [1. -1.]
# end
# function vcov(o::LinReg)
#     coef(o)
#     p = length(o.β)
#     -mse(o) * o.S[1:p, 1:p] / nobs(o)
#  end
# stderr(o::LinReg) = sqrt.(diag(vcov(o)))
#
# function Base.merge!(o1::LinReg, o2::LinReg, γ::Float64)
#     @assert o1.λ == o2.λ
#     @assert length(o1.β) == length(o2.β)
#     smooth!(o1.A, o2.A, γ)
#     o1.nobs += o2.nobs
#     coef(o1)
#     o1
# end

#-----------------------------------------------------------------------# Mean
"""
    Mean()
Univariate mean.

    s = Series(randn(100), Mean())
    value(s)
"""
mutable struct Mean <: OnlineStat{0, EqualWeight}
    μ::Float64
    Mean() = new(0.0)
end
fit!(o::Mean, y::Real, γ::Float64) = (o.μ = smooth(o.μ, y, γ))
Base.merge!(o::Mean, o2::Mean, γ::Float64) = (fit!(o, value(o2), γ); o)
Base.mean(o::Mean) = value(o)


#-----------------------------------------------------------------------# Moments
"""
    Moments()
First four non-central moments.

    s = Series(randn(1000), Moments(10))
    value(s)
"""
mutable struct Moments <: OnlineStat{0, EqualWeight}
    m::Vector{Float64}
    nobs::Int
    Moments() = new(zeros(4), 0)
end
function fit!(o::Moments, y::Real, γ::Float64)
    o.nobs += 1
    @inbounds o.m[1] = smooth(o.m[1], y, γ)
    @inbounds o.m[2] = smooth(o.m[2], y * y, γ)
    @inbounds o.m[3] = smooth(o.m[3], y * y * y, γ)
    @inbounds o.m[4] = smooth(o.m[4], y * y * y * y, γ)
end
Base.mean(o::Moments) = o.m[1]
Base.var(o::Moments) = (o.m[2] - o.m[1] ^ 2) * unbias(o)
Base.std(o::Moments) = sqrt.(var(o))
function skewness(o::Moments)
    v = value(o)
    (v[3] - 3.0 * v[1] * var(o) - v[1] ^ 3) / var(o) ^ 1.5
end
function kurtosis(o::Moments)
    v = value(o)
    (v[4] - 4.0 * v[1] * v[3] + 6.0 * v[1] ^ 2 * v[2] - 3.0 * v[1] ^ 4) / var(o) ^ 2 - 3.0
end
function Base.merge!(o1::Moments, o2::Moments, γ::Float64)
    smooth!(o1.m, o2.m, γ)
    o1.nobs += o2.nobs
    o1
end

#-----------------------------------------------------------------------# OHistogram
"""
    OHistogram(range)
Make a histogram with bins given by `range`.  Uses left-closed bins.

    y = randn(100)
    s = Series(y, OHistogram(-4:.1:4))
    value(s)
"""
struct OHistogram{H <: Histogram} <: OnlineStat{0, EqualWeight}
    h::H
end
OHistogram(r::Range) = OHistogram(Histogram(r, :left))
function fit!(o::OHistogram, y::ScalarOb, γ::Float64)
    H = o.h
    x = H.edges[1]
    a = first(x)
    δ = step(x)
    k = floor(Int, (y - a) / δ) + 1
    if 1 <= k <= length(x)
        @inbounds H.weights[k] += 1
    end
end

#-----------------------------------------------------------------------# OrderStats
"""
    OrderStats(b)
Average order statistics with batches of size `b`.

    s = Series(randn(1000), OrderStats(10))
    value(s)
"""
mutable struct OrderStats <: OnlineStat{0, EqualWeight}
    value::Vector{Float64}
    buffer::Vector{Float64}
    i::Int
    nreps::Int
    OrderStats(p::Integer) = new(zeros(p), zeros(p), 0, 0)
end
function fit!(o::OrderStats, y::Real, γ::Float64)
    p = length(o.value)
    buffer = o.buffer
    o.i += 1
    buffer[o.i] = y
    if o.i == p
        sort!(buffer)
        o.nreps += 1
        o.i = 0
        smooth!(o.value, buffer, 1 / o.nreps)
    end
    o
end

#-----------------------------------------------------------------------# QuantileMM
"""
    QuantileMM(q = 0.5)
Approximate quantiles via an online MM algorithm.

    s = Series(randn(1000), QuantileMM())
    value(s)
"""
struct QuantileMM <: OnlineStat{0, LearningRate}
    value::Vector{Float64}
    τ::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}
    QuantileMM(τ = [.25, .5, .75]) = new(zeros(τ), collect(τ), zeros(τ), zeros(τ))
end
function fit!(o::QuantileMM, y::Real, γ::Float64)
    γ == 1.0 && fill!(o.value, y)  # initialize values with first observation
    @inbounds for j in 1:length(o.τ)
        w = 1.0 / (abs(y - o.value[j]) + ϵ)
        o.s[j] = smooth(o.s[j], w * y, γ)
        o.t[j] = smooth(o.t[j], w, γ)
        o.value[j] = (o.s[j] + (2.0 * o.τ[j] - 1.0)) / o.t[j]
    end
end
function Base.merge!(o::QuantileMM, o2::QuantileMM, γ::Float64)
    o.τ == o2.τ || error("Objects track different quantiles")
    smooth!(o.value, o2.value, γ)
end

#-----------------------------------------------------------------------# QuantileMSPI
struct QuantileMSPI <: OnlineStat{0, LearningRate}
    value::Vector{Float64}
    τ::Vector{Float64}
    QuantileMSPI(τ = [.25, .5, .75]) = new(zeros(τ), collect(τ))
end
function fit!(o::QuantileMSPI, y::Real, γ::Float64)
    γ == 1.0 && fill!(o.value, y)  # initialize values with first observation
    @inbounds for i in eachindex(o.τ)
        w = inv(abs(y - o.value[i]) + ϵ)
        b = o.τ[i] - .5 * (1 - y * w)
        o.value[i] = (o.value[i] + γ * b) / (1 + .5 * γ * w)
    end
end
function Base.merge!(o::QuantileMSPI, o2::QuantileMSPI, γ::Float64)
    o.τ == o2.τ || error("Objects track different quantiles")
    smooth!(o.value, o2.value, γ)
end

#-----------------------------------------------------------------------# QuantileSGD
struct QuantileSGD <: OnlineStat{0, LearningRate}
    value::Vector{Float64}
    τ::Vector{Float64}
    QuantileSGD(τ = [.25, .5, .75]) = new(zeros(τ), collect(τ))
end
function fit!(o::QuantileSGD, y::Real, γ::Float64)
    γ == 1.0 && fill!(o.value, y)  # initialize values with first observation
    for j in eachindex(o.value)
        u = o.value[j] - y
        o.value[j] -= γ * ((u > 0.0) - o.τ[j])
    end
end
function Base.merge!(o::QuantileSGD, o2::QuantileSGD, γ::Float64)
    o.τ == o2.τ || error("Objects track different quantiles")
    smooth!(o.value, o2.value, γ)
end


#-----------------------------------------------------------------------# ReservoirSample
"""
    ReservoirSample(k, t = Float64)
Reservoir sample of `k` items.

    o = ReservoirSample(k, Int)
    s = Series(o)
    fit!(s, 1:10000)
"""
mutable struct ReservoirSample{T<:Number} <: OnlineStat{0, EqualWeight}
    value::Vector{T}
    nobs::Int
end
ReservoirSample{T<:Number}(k::Integer, ::Type{T} = Float64) = ReservoirSample(zeros(T, k), 0)

function fit!(o::ReservoirSample, y::ScalarOb, γ::Float64)
    o.nobs += 1
    if o.nobs <= length(o.value)
        o.value[o.nobs] = y
    else
        j = rand(1:o.nobs)
        if j < length(o.value)
            o.value[j] = y
        end
    end
end

#-----------------------------------------------------------------------# Sum
"""
    Sum()
Track the overall sum.

    s = Series(randn(1000), Sum())
    value(s)
"""
mutable struct Sum{T <: Real} <: OnlineStat{0, EqualWeight}
    sum::T
end
Sum() = Sum(0.0)
Sum{T<:Real}(::Type{T}) = Sum(zero(T))
Base.sum(o::Sum) = o.sum
fit!{T<:AbstractFloat}(o::Sum{T}, x::Real, γ::Float64) = (v = convert(T, x); o.sum += v)
fit!{T<:Integer}(o::Sum{T}, x::Real, γ::Float64) =       (v = round(T, x);   o.sum += v)
Base.merge!{T <: Sum}(o::T, o2::T, γ::Float64) = (o.sum += o2.sum)

#-----------------------------------------------------------------------# Variance
"""
    Variance()
Univariate variance.

    s = Series(randn(100), Variance())
    value(s)
"""
mutable struct Variance <: OnlineStat{0, EqualWeight}
    σ2::Float64     # biased variance
    μ::Float64
    nobs::Int
    Variance() = new(0.0, 0.0, 0)
end
function fit!(o::Variance, y::Real, γ::Float64)
    μ = o.μ
    o.nobs += 1
    o.μ = smooth(o.μ, y, γ)
    o.σ2 = smooth(o.σ2, (y - o.μ) * (y - μ), γ)
end
function Base.merge!(o::Variance, o2::Variance, γ::Float64)
    o.nobs += o2.nobs
    δ = o2.μ - o.μ
    o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    o.μ = smooth(o.μ, o2.μ, γ)
    o
end
value(o::Variance) = o.σ2 * unbias(o)
Base.var(o::Variance) = value(o)
Base.std(o::Variance) = sqrt(var(o))
Base.mean(o::Variance) = o.μ
nobs(o::Variance) = o.nobs
