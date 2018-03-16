#---------------------------------------------------------------------------------# Normal
"""
    FitNormal()

Calculate the parameters of a normal distribution via maximum likelihood.
"""
struct FitNormal{V <: Variance} <: OnlineStat{0}
    var::V
end
FitNormal(;kw...) = FitNormal(Variance(;kw...))
_fit!(o::FitNormal, y::Real) = _fit!(o.var, y)
nobs(o::FitNormal) = nobs(o.var)
function value(o::FitNormal)
    if nobs(o) > 1
        return mean(o.var), std(o.var)
    else
        return 0.0, 1.0
    end
end
Base.merge!(o::FitNormal, o2::FitNormal) = merge!(o.var, o2.var)
Base.mean(o::FitNormal) = mean(o.var)
Base.var(o::FitNormal) = var(o.var)

function pdf(o::FitNormal, x::Number) 
    σ = std(o)
    return 1 / (sqrt(2π) * σ) * exp(-(x - mean(o))^2 / 2σ^2)
end
cdf(o::FitNormal, x::Number) = .5 * (1.0 + erf((x - mean(o)) / (std(o) * √2)))