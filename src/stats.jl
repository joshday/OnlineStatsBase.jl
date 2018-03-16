#-----------------------------------------------------------------------# Bootstrap
"""
    Bootstrap(o::OnlineStat, nreps = 100, d = [0, 2])

Online statistical bootstrap.  Create `nreps` replicates of `o`.  For each call to `fit!`,
a replicate will be updated `rand(d)` times.

# Example

    o = Bootstrap(Variance())
    Series(randn(1000), o)
    confint(o)
"""
struct Bootstrap{N, O <: OnlineStat{N}, D} <: OnlineStat{N}
    stat::O 
    replicates::Vector{O}
    rnd::D
end
function Bootstrap(o::OnlineStat{N}, nreps::Integer = 100, d = [0, 2]) where {N}
    Bootstrap{N, typeof(o), typeof(d)}(o, [copy(o) for i in 1:nreps], d)
end
Base.show(io::IO, b::Bootstrap) = print(io, "Bootstrap($(length(b.replicates))): $(b.stat)")
"""
    confint(b::Bootstrap, coverageprob = .95)

Return a confidence interval for a Bootstrap `b`.
"""
function confint(b::Bootstrap, coverageprob = 0.95)
    states = value.(b.replicates)
    α = 1 - coverageprob
    return (quantile(states, α / 2), quantile(states, 1 - α / 2))
end
function fit_replicates!(b::Bootstrap, yi)
    for r in b.replicates
        for _ in 1:rand(b.rnd)
            _fit!(r, yi)
        end
    end
end
function _fit!(b::Bootstrap, y)
    _fit!(b.stat, y)
    fit_replicates!(b, y)
    b
end

#-----------------------------------------------------------------------# Count 
"""
    Count()

The number of things observed.

# Example 

    fit!(Count(), 1:1000)
"""
mutable struct Count <: OnlineStat{0}
    n::Int
    Count() = new(0)
end
_fit!(o::Count, x) = (o.n += 1)
Base.merge!(o::Count, o2::Count) = (o.n += o2.n; o)

#-----------------------------------------------------------------------# CountMap 
"""
    CountMap(T::Type)
    CountMap(dict::AbstractDict{T, Int})

Track a dictionary that maps unique values to its number of occurrences.  Similar to 
`StatsBase.countmap`.  

# Example 
    
    fit!(CountMap(Int), rand(1:10, 1000))
"""
struct CountMap{A <: AbstractDict} <: OnlineStat{0}
    value::A  # OrderedDict by default
end
CountMap(T::Type) = CountMap(OrderedDict{T, Int}())
_fit!(o::CountMap, x) = haskey(o.value, x) ? o.value[x] += 1 : o.value[x] = 1
Base.merge!(o::CountMap, o2::CountMap) = (merge!(+, o.value, o2.value); o)
nobs(o::CountMap) = sum(values(o.value))
function probs(o::CountMap, kys = keys(o.value))
    out = zeros(Int, length(kys))
    valkeys = keys(o.value)
    for (i, k) in enumerate(kys)
        out[i] = k in valkeys ? o.value[k] : 0
    end
    sum(out) == 0 ? Float64.(out) : out ./ sum(out)
end
pdf(o::CountMap, y) = y in keys(o.value) ? o.value[y] / nobs(o) : 0.0

#-----------------------------------------------------------------------# CovMatrix 
"""
    CovMatrix(p=0; weight)

Calculate a covariance/correlation matrix of `p` variables.  If the number of variables is 
unknown, leave the default `p=0`.

# Example 

    fit!(CovMatrix(), randn(100, 4))
"""
mutable struct CovMatrix{W} <: OnlineStat{1}
    value::Matrix{Float64}
    A::Matrix{Float64}  # x'x/n
    b::Vector{Float64}  # 1'x/n
    weight::W
    n::Int
end
CovMatrix(p::Int=0;weight = inv) = CovMatrix(zeros(p,p), zeros(p,p), zeros(p), weight, 0)
function _fit!(o::CovMatrix, x)
    γ = o.weight(o.n += 1)
    if isempty(o.A)
        p = length(x)
        o.b = Vector{Float64}(undef, p) 
        o.A = Matrix{Float64}(undef, p, p)
        o.value = Matrix{Float64}(undef, p, p)
    end
    smooth!(o.b, x, γ)
    smooth_syr!(o.A, x, γ)
end
function value(o::CovMatrix; corrected::Bool = true)
    o.value[:] = Matrix(Symmetric((o.A - o.b * o.b')))
    corrected && rmul!(o.value, unbias(o))
    o.value
end
function Base.merge!(o::CovMatrix, o2::CovMatrix)
    γ = o2.n / (o.n += o2.n)
    smooth!(o.A, o2.A, γ)
    smooth!(o.b, o2.b, γ)
    o
end
Base.cov(o::CovMatrix; corrected::Bool = true) = value(o; corrected=corrected)
Base.mean(o::CovMatrix) = o.b
Base.var(o::CovMatrix; kw...) = diag(value(o; kw...))
function Base.cor(o::CovMatrix; kw...)
    value(o; kw...)
    v = 1.0 ./ sqrt.(diag(o.value))
    rmul!(o.value, Diagonal(v))
    lmul!(Diagonal(v), o.value)
    o.value
end

#-----------------------------------------------------------------------# CStat
"""
    CStat(stat)

Track a univariate OnlineStat for complex numbers.  A copy of `stat` is made to
separately track the real and imaginary parts.

# Example
    
    y = randn(100) + randn(100)im
    fit!(y, CStat(Mean()))
"""
struct CStat{O <: OnlineStat{0}} <: OnlineStat{0}
    re_stat::O
    im_stat::O
end
CStat(o::OnlineStat{0}) = CStat(o, copy(o))
nobs(o::CStat) = nobs(o.re_stat)
value(o::CStat) = value(o.re_stat), value(o.im_stat)
_fit!(o::CStat, y::T) where {T<:Real} = (_fit!(o.re_stat, y); _fit!(o.im_stat, T(0)))
_fit!(o::CStat, y::Complex) = (_fit!(o.re_stat, y.re); _fit!(o.im_stat, y.im))
function Base.merge!(o::T, o2::T) where {T<:CStat}
    merge!(o.re_stat, o2.re_stat)
    merge!(o.im_stat, o2.im_stat)
end

#-----------------------------------------------------------------------# Diff
"""
    Diff(T::Type = Float64)

Track the difference and the last value.

# Example

    o = Diff()
    fit!(o, [1.0, 2.0])
    last(o)
    diff(o)
"""
mutable struct Diff{T <: Real} <: OnlineStat{0}
    diff::T
    lastval::T
end
Diff(T::Type = Float64) = Diff(zero(T), zero(T))
function _fit!(o::Diff{T}, x) where {T<:AbstractFloat}
    v = convert(T, x)
    o.diff = v - last(o)
    o.lastval = v
end
function _fit!(o::Diff{T}, x) where {T<:Integer}
    v = round(T, x)
    o.diff = v - last(o)
    o.lastval = v
end
Base.last(o::Diff) = o.lastval
Base.diff(o::Diff) = o.diff


#-----------------------------------------------------------------------# Extrema
"""
    Extrema(T::Type = Float64)

Maximum and minimum.

# Example

    fit!(Extrema(), rand(10^5))
"""
mutable struct Extrema{T} <: OnlineStat{0}
    min::T
    max::T
    n::Int
end
Extrema(T::Type = Float64) = Extrema{T}(typemax(T), typemin(T), 0)
function _fit!(o::Extrema, y::Real)
    o.min = min(o.min, y)
    o.max = max(o.max, y)
    o.n += 1
end
function Base.merge!(o::Extrema, o2::Extrema)
    o.min = min(o.min, o2.min)
    o.max = max(o.max, o2.max)
    o.n += o2.n
    o
end
value(o::Extrema) = (o.min, o.max)
Base.extrema(o::Extrema) = value(o)
Base.maximum(o::Extrema) = o.max 
Base.minimum(o::Extrema) = o.min

#-----------------------------------------------------------------------# FTSeries 
"""
    FTSeries(stats...; filter=always, transform=identity)

A series that filters and transforms the data before being fit.

# Example 

    fit!(FTSeries(Mean(), Variance(); transform=abs), -rand(1000))
"""
mutable struct FTSeries{N, OS<:Tup, F, T} <: OnlineStat{N}
    stats::OS
    filter::F 
    transform::T 
    nfiltered::Int
end
function FTSeries(stats::OnlineStat{N}...; filter=always, transform=identity) where {N}
    FTSeries{N, typeof(stats), typeof(filter), typeof(transform)}(stats, filter, transform, 0)
end
@generated function _fit!(o::FTSeries{N, OS}, y) where {N, OS}
    N = length(fieldnames(OS))
    quote
        Base.Cartesian.@nexprs $N i -> @inbounds begin
            yi = y[i]; o.filter(yi) ? _fit!(o.stats[i], o.transform(yi)) : o.nfiltered += 1
        end
    end
end
function Base.merge!(o::T, o2::T) where {T<:FTSeries}
    o.nfiltered += o2.nfiltered 
    merge!.(o.stats, o2.stats)
    o
end


always(x) = true

#-----------------------------------------------------------------------# HyperLogLog
# Mostly copy/pasted from StreamStats.jl
"""
    HyperLogLog(b)  # 4 ≤ b ≤ 16

Approximate count of distinct elements.

# Example

    fit!(HyperLogLog(12), rand(1:10,10^5))
"""
mutable struct HyperLogLog <: OnlineStat{0}
    m::UInt32
    M::Vector{UInt32}
    mask::UInt32
    altmask::UInt32
    n::Int
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
        new(m, M, mask, altmask, 0)
    end
end
function Base.show(io::IO, o::HyperLogLog)
    print(io, "HyperLogLog($(o.m) registers, estimate = $(value(o)))")
end

hash32(d::Any) = hash(d) % UInt32
maskadd32(x::UInt32, mask::UInt32, add::UInt32) = (x & mask) + add
ρ(s::UInt32) = UInt32(leading_zeros(s)) + 0x00000001

function α(m::UInt32)
    if m == 0x00000010          # m = 16
        return 0.673
    elseif m == 0x00000020      # 
        return 0.697
    elseif m == 0x00000040
        return 0.709
    else                        # if m >= UInt32(128)
        return 0.7213 / (1 + 1.079 / m)
    end
end

function _fit!(o::HyperLogLog, v)
    o.n += 1
    x = hash32(v)
    j = maskadd32(x, o.mask, 0x00000001)
    w = x & o.altmask
    o.M[j] = max(o.M[j], ρ(w))
    o
end

function value(o::HyperLogLog)
    S = 0.0
    for j in eachindex(o.M)
        S += 1 / (2 ^ o.M[j])
    end
    Z = 1 / S
    E = α(o.m) * UInt(o.m) ^ 2 * Z
    if E <= 5//2 * o.m
        V = 0
        for j in 1:o.m
            V += Int(o.M[j] == 0x00000000)
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

function Base.merge!(o::HyperLogLog, o2::HyperLogLog)
    length(o.M) == length(o2.M) || 
        error("Merge failed. HyperLogLog objects have different number of registers.")
    o.n += o2.n
    for j in eachindex(o.M)
        o.M[j] = max(o.M[j], o2.M[j])
    end
    o
end


#-----------------------------------------------------------------------# Mean
"""
    Mean(; weight)

Track a univariate mean.

# Update 

``μ = (1 - w) * μ + w * x``

# Example

    @time fit!(Mean(), randn(10^6))

    # exponentially-weighted mean
    @time fit!(Mean(;weight = x -> 0.1), randn(10^6))
"""
mutable struct Mean{W} <: OnlineStat{0}
    μ::Float64
    weight::W
    n::Int
end
Mean(;weight = EqualWeight()) = Mean(0.0, weight, 0)
_fit!(o::Mean, x) = (o.μ = smooth(o.μ, x, o.weight(o.n += 1)))
function Base.merge!(o::Mean, o2::Mean) 
    o.n += o2.n
    o.μ = smooth(o.μ, o2.μ, o2.n / o.n)
    o
end
Base.mean(o::Mean) = o.μ

#-----------------------------------------------------------------------# ProbMap
"""
    ProbMap(T::Type; weight)
    ProbMap(A::AbstractDict; weight)

Track a dictionary that maps unique values to its probability.  Similar to 
[`CountMap`](@ref), but uses a weighting mechanism.

# Example 
    
    fit!(ProbMap(Int), rand(1:10, 1000))
"""
mutable struct ProbMap{A<:AbstractDict, W} <: OnlineStat{0}
    value::A 
    weight::W 
    n::Int
end
ProbMap(T::Type; weight = inv) = ProbMap(OrderedDict{T, Float64}(), weight, 0)
function _fit!(o::ProbMap, y)
    γ = o.weight(o.n += 1)
    get!(o.value, y, 0.0)   # initialize class probability at 0 if it isn't present
    for ky in keys(o.value)
        if ky == y 
            o.value[ky] = smooth(o.value[ky], 1.0, γ)
        else 
            o.value[ky] *= (1 - γ)
        end
    end
end
function Base.merge!(o::ProbMap, o2::ProbMap) 
    o.n += o2.n
    merge!((a, b)->smooth(a, b, o.n2 / o.n), o.value, o2.value)
    o
end
function probs(o::ProbMap, levels = keys(o))
    out = zeros(length(levels))
    for (i, ky) in enumerate(levels)
        out[i] = get(o.value, ky, 0.0)
    end
    sum(out) == 0.0 ? out : out ./ sum(out)
end

#-----------------------------------------------------------------------# Series
"""
    Series(stats...)

Track multiple stats for one data stream.

# Example 

    s = Series(Mean(), Variance())
    fit!(s, randn(1000))
"""
struct Series{N, T<:Tup} <: OnlineStat{N}
    stats::T
end
Series(stats::OnlineStat{N}...) where {N} = Series{N, typeof(stats)}(stats)
@generated function _fit!(o::Series{N, T}, y) where {N, T}
    N = length(fieldnames(T))
    :(Base.Cartesian.@nexprs $N i -> @inbounds(_fit!(o.stats[i], y[i])))
end
Base.merge!(o::Series, o2::Series) = (merge!.(o.stats, o2.stats); o)

#-----------------------------------------------------------------------# Variance 
"""
    Variance(; weight)

Univariate variance.

# Example 

    @time fit!(Variance(), randn(10^6))
"""
mutable struct Variance{W} <: OnlineStat{0}
    σ2::Float64 
    μ::Float64 
    weight::W
    n::Int
end
Variance(;weight = inv) = Variance(0.0, 0.0, weight, 0)
function _fit!(o::Variance, x)
    μ = o.μ
    γ = o.weight(o.n += 1)
    o.μ = smooth(o.μ, x, γ)
    o.σ2 = smooth(o.σ2, (x - o.μ) * (x - μ), γ)
end
function Base.merge!(o::Variance, o2::Variance)
    γ = o2.n / (o.n += o2.n)
    δ = o2.μ - o.μ
    o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    o.μ = smooth(o.μ, o2.μ, γ)
    o
end
value(o::Variance) = o.n > 0 ? o.σ2 * unbias(o) : 0.0
Base.var(o::Variance) = value(o)
Base.mean(o::Variance) = o.μ

#-----------------------------------------------------------------------# AutoCov and Lag
"""
    Lag(b, T = Float64)

Store the last `b` values for a data stream of type `T`.
"""
struct Lag{T} <: OnlineStat{0}
    value::Vector{T}
end
Lag(b::Integer, T::Type = Float64) = Lag(zeros(T, b))
function _fit!(o::Lag{T}, y::T) where {T} 
    for i in reverse(2:length(o.value))
        @inbounds o.value[i] = o.value[i - 1]
    end
    o.value[1] = y
end

"""
    AutoCov(b, T = Float64)

Calculate the auto-covariance/correlation for lags 0 to `b` for a data stream of type `T`.

# Example 

    y = cumsum(randn(100))
    o = AutoCov(5)
    fit!(o, y)
    autocov(o)
    autocor(o)
"""
struct AutoCov{T, W} <: OnlineStat{0}
    cross::Vector{Float64}
    m1::Vector{Float64}
    m2::Vector{Float64}
    lag::Lag{T}         # y_{t-1}, y_{t-2}, ...
    wlag::Lag{Float64}  # γ_{t-1}, γ_{t-2}, ...
    v::Variance{W}
end
function AutoCov(k::Integer, T = Float64; kw...)
    d = k + 1
    AutoCov(zeros(d), zeros(d), zeros(d), Lag(d, T), Lag(d, Float64), Variance(;kw...))
end
nobs(o::AutoCov) = nobs(o.v)

function _fit!(o::AutoCov, y::Real)
    γ = o.v.weight(o.v.n + 1)
    _fit!(o.v, y)
    _fit!(o.lag, y)     # y_t, y_{t-1}, ...
    _fit!(o.wlag, γ)    # γ_t, γ_{t-1}, ...
    # M1 ✓
    for k in reverse(2:length(o.m2))
        @inbounds o.m1[k] = o.m1[k - 1]
    end
    @inbounds o.m1[1] = smooth(o.m1[1], y, γ)
    # Cross ✓ and M2 ✓
    @inbounds for k in 1:length(o.m1)
        γk = value(o.wlag)[k]
        o.cross[k] = smooth(o.cross[k], y * value(o.lag)[k], γk)
        o.m2[k] = smooth(o.m2[k], y, γk)
    end
end

function value(o::AutoCov)
    μ = mean(o.v)
    n = nobs(o)
    cr = o.cross
    m1 = o.m1
    m2 = o.m2
    [(n - k + 1) / n * (cr[k] + μ * (μ - m1[k] - m2[k])) for k in 1:length(m1)]
end
autocov(o::AutoCov) = value(o)
autocor(o::AutoCov) = value(o) ./ value(o)[1]
