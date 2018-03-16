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
struct Series{N, T<:Tup} <: OnlineStat{N}
    stats::T
end
Series(stats::OnlineStat{N}...) where {N} = Series{N, typeof(stats)}(stats)
@generated function _fit!(o::Series{N, T}, y) where {N, T}
    N = length(fieldnames(T))
    :(Base.Cartesian.@nexprs $N i -> @inbounds(_fit!(o.stats[i], y[i])))
end

#-----------------------------------------------------------------------# FTSeries 
struct FTSeries{N, OS<:Tup, F, T} <: OnlineStat{N}
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
            yi = y[i]; o.filter(yi) && _fit!(o.stats[i], o.transform(yi))
        end
    end
end

always(x) = true