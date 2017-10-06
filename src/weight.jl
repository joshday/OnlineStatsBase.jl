#============================================================================= Weight
Subtypes of Weight need at least the fields
- `nobs`
- `nups`
and a method for
- `weight(w, n2::Int = 1)`
==============================================================================#
abstract type Weight end

# Base functions
Base.show(io::IO, w::Weight) = (print(io, name(w)); show_fields(io, w))
fields_to_show(w::Weight) = setdiff(fieldnames(w), [:nups])
function Base.:(==){T <: Weight}(w1::T, w2::T)
    nms = fieldnames(w1)
    all(getfield.(w1, nms) .== getfield.(w2, nms))
end
Base.copy(w::Weight) = deepcopy(w)

# interface
nobs(w::Weight) = w.nobs
nups(w::Weight) = w.nups
updatecounter!(w::Weight, n2::Int = 1) = (w.nobs += n2;w.nups += 1)
weight!(w::Weight, n2::Int = 1) = (updatecounter!(w, n2); weight(w, n2))
weight(w::Weight, n2::Int = 1) = error("$w has not defined the required `weight(w, n2=1)` method")


#-------------------------------------------------------------------------# Bounded
"""
    Bounded(weight, λ)
Give a Weight a lower bound.
"""
struct Bounded{W <: Weight} <: Weight
    w::W
    λ::Float64
    Bounded(w::W, λ::Float64 = .05) where W<:Weight = new{W}(w, λ)
end
nobs(w::Bounded) = nobs(w.w)
nups(w::Bounded) = nups(w.w)
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
nups(w::Scaled) = nups(w.w)
updatecounter!(w::Scaled, n2::Int = 1) = updatecounter!(w.w, n2)
weight(w::Scaled, n2::Int = 1) = weight(w.w, n2) * w.λ
Base.show(io::IO, w::Scaled) = print(io, "$(w.λ) * $(w.w)")


#-------------------------------------------------------------------------# EqualWeight
"""
    EqualWeight()
- Equally weighted observations
- Weight at observation `t` is `γ = 1 / t`
"""
mutable struct EqualWeight <: Weight
    nobs::Int
    nups::Int
    EqualWeight() = new(0, 0)
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
    nups::Int
    ExponentialWeight(λ::Real = 0.1) = new(λ, 0, 0)
    ExponentialWeight(lookback::Integer) = new(2 / (lookback + 1), 0, 0)
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
    nups::Int
    LearningRate(r::Real = .6) = new(r, 0, 0)
end
weight(w::LearningRate, n2::Int = 1) = exp(-w.r * log(w.nups))
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
    nups::Int
    LearningRate2(c::Real = 0.5) = new(c, 0, 0)
end
function weight(w::LearningRate2, n2::Int = 1)
    1.0 / (1.0 + w.c * (w.nups - 1))
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
    nups::Int
    function HarmonicWeight(a::Float64 = 10.0)
        a > 0 || throw(ArgumentError("`a` must be greater than 0"))
        new(a, 0, 0)
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
    nups::Int
    function McclainWeight(α = .1)
        0 < α < 1 || throw(ArgumentError("value must be between 0 and 1"))
        new(Float64(α), 1.0, 0, 0)
    end
end
fields_to_show(w::McclainWeight) = [:α, :nobs]
function weight(w::McclainWeight, n2::Int = 1)
    w.nups == 1 && return 1.0
    w.last = w.last / (1 + w.last - w.α)
end
