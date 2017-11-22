Base.show(io::IO, w::Weight) = print(io, name(w))
function Base.:(==)(o1::Weight, o2::Weight)
    typeof(o1) == typeof(o2) || return false
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end
Base.copy(w::Weight) = deepcopy(w)

#-----------------------------------------------------------------------# EqualWeight
"""
    EqualWeight()

Equally weighted observations.  

``γ(t) = 1 / t``

# Example

    Series(randn(100), EqualWeight(), Variance())
"""
struct EqualWeight <: Weight end
(::EqualWeight)(n, n2 = 1) = n2 / n

#-----------------------------------------------------------------------# ExponentialWeight
"""
    ExponentialWeight(λ::Float64)
    ExponentialWeight(lookback::Int)

Exponentially weighted observations.  

``γ(t) = λ = 2 / (lookback + 1)``

# Example

    Series(randn(100), ExponentialWeight(), Variance())
"""
struct ExponentialWeight <: Weight 
    λ::Float64 
    ExponentialWeight(λ::Real = .1) = new(λ)
    ExponentialWeight(lookback::Integer) = new(2 / (lookback + 1))
end
(w::ExponentialWeight)(n, n2=1) = n == 1 ? 1.0 : w.λ
Base.show(io::IO, w::ExponentialWeight) = print(io, name(w) * "(λ = $(w.λ))")

#-----------------------------------------------------------------------# LearningRate
"""
    LearningRate(r = .6)

Slowly decreasing weight.  

``γ(t) = 1 / t^r``

# Example

    Series(randn(1000), LearningRate(.7), QuantileMM(), QuantileMSPI(), QuantileSGD())
"""
struct LearningRate <: Weight 
    r::Float64 
    LearningRate(r = .6) = new(r)
end
(w::LearningRate)(n, n2=1) = 1 / n ^ w.r
Base.show(io::IO, w::LearningRate) = print(io, name(w) * "(r = $(w.r))")

#-----------------------------------------------------------------------# LearningRate2
"""
    LearningRate2(c = .5)

Slowly decreasing weight.  

``γ(t) = 1 / (1 + c * (t - 1))``

# Example

    Series(randn(1000), LearningRate2(.3), QuantileMM(), QuantileMSPI(), QuantileSGD())
"""
struct LearningRate2 <: Weight 
    c::Float64 
    LearningRate2(c = .5) = new(c)
end
(w::LearningRate2)(n, n2=1) = 1 / (1 + w.c * (n - 1))
Base.show(io::IO, w::LearningRate2) = print(io, name(w) * "(c = $(w.c))")

#-----------------------------------------------------------------------# HarmonicWeight
"""
    HarmonicWeight(a = 10.0)

Weight determined by harmonic series.  

``γ(t) = a / (a + t - 1)``

# Example

    Series(randn(1000), HarmonicWeight(), QuantileMSPI())
"""
struct HarmonicWeight <: Weight 
    a::Float64 
    HarmonicWeight(a = 10.0) = new(a)
end
(w::HarmonicWeight)(n, n2=1) = w.a / (w.a + n - 1)
Base.show(io::IO, w::HarmonicWeight) = print(io, name(w) * "(a = $(w.a))")

#-----------------------------------------------------------------------# McclainWeight
"""
    McclainWeight(α = .1)

Weight which decreases into a constant.

``γ(t) = γ(t-1) / (1 + γ(t) - α)``

# Example

    Series(randn(100), McclainWeight(), Mean())
"""
mutable struct McclainWeight <: Weight
    α::Float64
    last::Float64
    McclainWeight(α = .1) = new(α, 1.0)
end
(w::McclainWeight)(n, n2=1) = n == 1 ? 1.0 : (w.last = w.last / (1 + w.last - w.α))
Base.show(io::IO, w::McclainWeight) = print(io, name(w) * "(α = $(w.α))")

#-----------------------------------------------------------------------# Bounded
"""
    Bounded(w::Weight, λ::Float64)

Bound the weight by a constant.

``γ_{bounded}(t) = max(γ(t), λ)``

# Example

    Bounded(EqualWeight(), .1)
"""
struct Bounded{W <: Weight} <: Weight 
    weight::W 
    λ::Float64 
end
(w::Bounded)(n, n2=1) = max(w.λ, w.weight(n, n2))
Base.show(io::IO, w::Bounded) = print(io, "max($(w.λ), $(w.weight))")
Base.max(w::Weight, λ::Float64) = Bounded(w, λ)
Base.max(λ::Float64, w::Weight) = Bounded(w, λ)

#-----------------------------------------------------------------------# Scaled
"""
    Scaled(w::Weight, λ::Float64)

Scale a weight by a constant.

``γ_{scaled}(t) = λ * γ(t)``

# Example

    Bounded(LearningRate(.5), .1)

    Series(randn(1000), 2.0 * LearningRate(.9), QuantileMM())
"""
struct Scaled{W <: Weight} <: Weight
    weight::W 
    λ::Float64
end
Base.:*(λ::Real, w::Weight) = Scaled(w, Float64(λ))
Base.:*(w::Weight, λ::Real) = Scaled(w, Float64(λ))
(w::Scaled)(n, n2=1) = w.λ * w.weight(n, n2)
Base.show(io::IO, w::Scaled) = print(io, "$(w.λ) * $(w.weight)")
