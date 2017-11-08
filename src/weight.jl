module Weight

import ..AbstractWeight 

export Equal, Exponential, LearningRate, LearningRate2, Harmonic, Mcclain,
    Bounded, Scaled

#-----------------------------------------------------------------------# Equal
"""
    Weight.Equal()

Equally weighted observations: ``\gamma_t = 1 / t``
"""
struct Equal <: AbstractWeight end
(::Equal)(n) = 1 / n

#-----------------------------------------------------------------------# Exponential
struct Exponential <: AbstractWeight 
    λ::Float64 
    Exponential(λ::Real = .1) = new(λ)
    Exponential(lookback::Integer) = new(2 / (lookback + 1))
end
(w::Exponential)(n) = n == 1 ? 1.0 : w.λ
#-----------------------------------------------------------------------# LearningRate
struct LearningRate <: AbstractWeight 
    r::Float64 
    LearningRate(r = .6) = new(r)
end
(w::LearningRate)(n) = 1 / n ^ w.r
#-----------------------------------------------------------------------# LearningRate2
struct LearningRate2 <: AbstractWeight 
    c::Float64 
    LearningRate2(c = .5) = new(c)
end
(w::LearningRate2)(n) = 1 / (1 + w.c * (n - 1))
#-----------------------------------------------------------------------# Harmonic
struct Harmonic <: AbstractWeight 
    a::Float64 
    Harmonic(a = 10.0) = new(a)
end
(w::Harmonic)(n) = w.a / (w.a + n - 1)
#-----------------------------------------------------------------------# Mcclain
mutable struct Mcclain <: AbstractWeight
    α::Float64
    last::Float64
    Mcclain(α = .1) = new(α, 1.0)
end
(w::Mcclain)(n) = n == 1 ? 1.0 : (w.last = w.last / (1 + w.last - w.α))
#-----------------------------------------------------------------------# Bounded
struct Bounded{W <: AbstractWeight} <: AbstractWeight 
    weight::W 
    λ::Float64 
end
(w::Bounded)(n) = max(w.λ, w.weight(n))
#-----------------------------------------------------------------------# Scaled
struct Scaled{W <: AbstractWeight} <: AbstractWeight
    weight::W 
    λ::Float64
end
Base.:*(λ::Real, w::AbstractWeight) = Scaled(w, Float64(λ))
(w::Scaled)(n) = w.λ * w.weight(n)
end