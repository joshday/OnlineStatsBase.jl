#---------------------------------------------------------------------------------# Normal
"""
    FitNormal()

Calculate the parameters of a normal distribution via maximum likelihood.
"""
struct FitNormal{V <: Variance} <: OnlineStat{0}
    v::V
end
FitNormal(;kw...) = FitNormal(Variance(;kw...))
_fit!(o::FitNormal, y::Real) = _fit!(o.v, y)
nobs(o::FitNormal) = nobs(o.v)
function value(o::FitNormal)
    if nobs(o) > 1
        return mean(o.v), std(o.v)
    else
        return 0.0, 1.0
    end
end
Base.merge!(o::FitNormal, o2::FitNormal) = (merge!(o.v, o2.v); o)
Base.mean(o::FitNormal) = mean(o.v)
Base.var(o::FitNormal) = var(o.v)

function pdf(o::FitNormal, x::Number) 
    σ = std(o)
    return 1 / (sqrt(2π) * σ) * exp(-(x - mean(o))^2 / 2σ^2)
end
cdf(o::FitNormal, x::Number) = .5 * (1.0 + erf((x - mean(o)) / (std(o) * √2)))