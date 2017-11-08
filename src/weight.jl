Base.show(io::IO, w::Weight) = print(io, name(w))

#-----------------------------------------------------------------------# EqualWeight
struct EqualWeight <: Weight end
(::EqualWeight)(n) = 1 / n

#-----------------------------------------------------------------------# ExponentialWeight
struct ExponentialWeight <: Weight 
    λ::Float64 
    ExponentialWeight(λ::Real = .1) = new(λ)
    ExponentialWeight(lookback::Integer) = new(2 / (lookback + 1))
end
(w::ExponentialWeight)(n) = n == 1 ? 1.0 : w.λ
#-----------------------------------------------------------------------# LearningRate
struct LearningRate <: Weight 
    r::Float64 
    LearningRate(r = .6) = new(r)
end
(w::LearningRate)(n) = 1 / n ^ w.r
#-----------------------------------------------------------------------# LearningRate2
struct LearningRate2 <: Weight 
    c::Float64 
    LearningRate2(c = .5) = new(c)
end
(w::LearningRate2)(n) = 1 / (1 + w.c * (n - 1))
#-----------------------------------------------------------------------# HarmonicWeight
struct HarmonicWeight <: Weight 
    a::Float64 
    HarmonicWeight(a = 10.0) = new(a)
end
(w::HarmonicWeight)(n) = w.a / (w.a + n - 1)
#-----------------------------------------------------------------------# McclainWeight
mutable struct McclainWeight <: Weight
    α::Float64
    last::Float64
    McclainWeight(α = .1) = new(α, 1.0)
end
(w::McclainWeight)(n) = n == 1 ? 1.0 : (w.last = w.last / (1 + w.last - w.α))
#-----------------------------------------------------------------------# Bounded
struct Bounded{W <: Weight} <: Weight 
    weight::W 
    λ::Float64 
end
(w::Bounded)(n) = max(w.λ, w.weight(n))
#-----------------------------------------------------------------------# Scaled
struct Scaled{W <: Weight} <: Weight
    weight::W 
    λ::Float64
end
Base.:*(λ::Real, w::Weight) = Scaled(w, Float64(λ))
(w::Scaled)(n) = w.λ * w.weight(n)
