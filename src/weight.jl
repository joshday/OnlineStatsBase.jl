# A Weight needs the field `nobs` and a method `weight(w, n2 = 1)`
abstract type Weight end

Base.show(io::IO, w::Weight) = (print(io, name(w)); show_fields(io, w))

function Base.:(==){T <: Weight}(w1::T, w2::T)
    nms = fieldnames(w1)
    all(getfield.(w1, nms) .== getfield.(w2, nms))
end

Base.copy(w::Weight) = deepcopy(w)

nobs(w::Weight) = w.nobs

updatecounter!(w::Weight, n2::Int = 1) = (w.nobs += n2)

weight!(w::Weight, n2::Int = 1) = (updatecounter!(w, n2); weight(w, n2))

#-----------------------------------------------------------------------# Bounded
"""
    Bounded(weight, λ)

Give a Weight a lower bound.
"""
struct Bounded{W <: Weight} <: Weight
    w::W
    λ::Float64
    Bounded(w::W, λ::Float64 = .05) where {W<:Weight} = new{W}(w, λ)
end
nobs(w::Bounded) = nobs(w.w)
updatecounter!(w::Bounded, n2::Int = 1) = updatecounter!(w.w, n2)
weight(w::Bounded, n2::Int = 1) = max(weight(w.w, n2), w.λ)
Base.show(io::IO, w::Bounded) = print(io, "Bounded by $(w.λ): $(w.w)")


#-----------------------------------------------------------------------# Scaled
"""
    Scaled(weight, λ)
    λ * weight

Scale a weight by a constant.
"""
struct Scaled{W <: Weight} <: Weight
    w::W
    λ::Float64
end
Base.:*(λ::Real, w::Weight) = Scaled(w, Float64(λ))
nobs(w::Scaled) = nobs(w.w)
updatecounter!(w::Scaled, n2::Int = 1) = updatecounter!(w.w, n2)
weight(w::Scaled, n2::Int = 1) = weight(w.w, n2) * w.λ
Base.show(io::IO, w::Scaled) = print(io, "$(w.λ) * $(w.w)")

#-----------------------------------------------------------------------#
#                                                                 Weight
#-----------------------------------------------------------------------#
#-------------------------------------------------------------------------# EqualWeight
"""
    EqualWeight()

- Equally weighted observations
- Weight at observation `t` is `γ = 1 / t`
"""
mutable struct EqualWeight <: Weight
    nobs::Int
    EqualWeight() = new(0)
end
weight(w::EqualWeight, n2::Int = 1) = n2 / w.nobs
#-------------------------------------------------------------------------# ExponentialWeight
"""
    ExponentialWeight(λ::Real = 0.1)
    ExponentialWeight(lookback::Integer)

- Exponentially weighted observations (constant)
- Weight at observation `t` is `γ = λ`
"""
mutable struct ExponentialWeight <: Weight
    λ::Float64
    nobs::Int
    ExponentialWeight(λ::Real = 0.1) = new(λ, 0)
    ExponentialWeight(lookback::Integer) = new(2 / (lookback + 1), 0)
end
weight(w::ExponentialWeight, n2::Int = 1) = w.λ
#-------------------------------------------------------------------------# LearningRate
"""
    LearningRate(r = .6)

- Mainly for stochastic approximation types
- Decreases at a "slow" rate
- Weight at observation `t` is `γ = 1 / t ^ r`
"""
mutable struct LearningRate <: Weight
    r::Float64
    nobs::Int
    LearningRate(r::Real = .6) = new(r, 0)
end
weight(w::LearningRate, n2::Int = 1) = exp(-w.r * log(nobs(w)))
#-------------------------------------------------------------------------# LearningRate2
"""
    LearningRate2(c = .5)

- Mainly for stochastic approximation types
- Decreases at a "slow" rate
- Weight at observation `t` is `γ = inv(1 + c * (t - 1))`
"""
mutable struct LearningRate2 <: Weight
    c::Float64
    nobs::Int
    LearningRate2(c::Real = 0.5) = new(c, 0)
end
function weight(w::LearningRate2, n2::Int = 1)
    1.0 / (1.0 + w.c * (nobs(w) - 1))
end
#-------------------------------------------------------------------------# HarmonicWeight
"""
    HarmonicWeight(a = 10.0)

- Decreases at a slow rate
- Weight at observation `t` is `γ = a / (a + t - 1)`
"""
mutable struct HarmonicWeight <: Weight
    a::Float64
    nobs::Int
    function HarmonicWeight(a::Real = 10.0)
        a > 0 || throw(ArgumentError("`a` must be greater than 0"))
        new(a, 0)
    end
end
function weight(w::HarmonicWeight, n2::Int = 1)
    w.a / (w.a + w.nobs - 1)
end
#-------------------------------------------------------------------------# McclainWeight
# Link with many weighting schemes:
# http://castlelab.princeton.edu/ORF569papers/Powell%20ADP%20Chapter%206.pdf
"""
    McclainWeight(ᾱ = 0.1)

- "smoothed" version of `BoundedEqualWeight`
- weights asymptotically approach `ᾱ`
- Weight at observation `t` is `γ(t-1) / (1 + γ(t-1) - ᾱ)`
"""
mutable struct McclainWeight <: Weight
    α::Float64
    last::Float64
    nobs::Int
    function McclainWeight(α = .1)
        0 < α < 1 || throw(ArgumentError("value must be between 0 and 1"))
        new(Float64(α), 1.0, 0)
    end
end
fields_to_show(w::McclainWeight) = [:α, :nobs]
function weight(w::McclainWeight, n2::Int = 1)
    nobs(w) == 1 && return 1.0
    w.last = w.last / (1 + w.last - w.α)
end
