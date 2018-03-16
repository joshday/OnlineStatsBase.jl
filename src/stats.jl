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
