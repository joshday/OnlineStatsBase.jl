abstract type Algorithm end 
nobs(o::Algorithm) = o.n
abstract type SGAlgorithm <: Algorithm end

#-----------------------------------------------------------------------# SGD
mutable struct SGD{W} <: SGAlgorithm
    δ::Vector{Float64}
    weight::W 
    n::Int
end
SGD(p=0; rate = LearningRate()) = SGD(zeros(p), rate, 0)
init!(o::SGD, p) = (o.δ = zeros(p))
function direction!(o::SGD) 
    γ = o.weight(o.n)
    for i in eachindex(o.δ)
        o.δ[i] = γ * o.δ[i]
    end
end
Base.merge!(o::SGD, o2::SGD) = (o.n += o2.n; o)

#-----------------------------------------------------------------------# ADAGRAD 
mutable struct ADAGRAD{W} <: SGAlgorithm 
    δ::Vector{Float64}
    weight::W 
    n::Int
    h::Vector{Float64}
end
ADAGRAD(p=0; rate=LearningRate()) = ADAGRAD(zeros(p), rate, 0, zeros(p))
init!(o::ADAGRAD, p) = (o.δ = zeros(p); o.h = zeros(p))
function direction!(o::ADAGRAD)
    γ = o.weight(o.n)
    for i in eachindex(o.h)
        o.h[i] = smooth(o.h[i], o.δ[i] ^ 2, γ)
        o.δ[i] = γ * o.δ[i] / (o.h[i] + ϵ)
    end
end
function Base.merge!(o::ADAGRAD, o2::ADAGRAD)
    o.n += o2.n 
    smooth!(o.h, o2.h, nobs(o2) / nobs(o))
    o
end

#-----------------------------------------------------------------------# ADADELTA 
mutable struct ADADELTA{W} <: SGAlgorithm 
    δ::Vector{Float64}
    weight::W 
    n::Int 
    v::Vector{Float64}
    Δ::Vector{Float64}
    ρ::Float64
end
ADADELTA(p=0; rate=LearningRate(), ρ = .95) = ADADELTA(zeros(p), rate, 0, zeros(p), zeros(p), ρ)
init!(o::ADADELTA, p) = (o.δ = zeros(p); o.v = zeros(p); o.Δ = zeros(p))
function direction!(o::ADADELTA)
    for i in eachindex(o.δ)
        g = o.δ[i]
        o.v[i] = smooth(g * g, o.v[i], o.ρ)
        Δi = (sqrt(o.Δ[i]) + ϵ) / (sqrt(o.v[i]) + ϵ)
        o.δ[i] *= Δi
        o.Δ[i] = smooth(Δi * Δi, o.Δ[i], o.ρ)
    end
end

#-----------------------------------------------------------------------# ADAM 
mutable struct ADAM{W} <: SGAlgorithm 
    δ::Vector{Float64}
    weight::W 
    n::Int
    m::Vector{Float64}
    v::Vector{Float64}
    β1::Float64 
    β2::Float64
end
function ADAM(p=0; rate=LearningRate(), β1 = .99, β2 = .999)
    ADAM(zeros(p), rate, 0, zeros(p), zeros(p), β1, β2)
end
init!(o::ADAM, p) = (o.δ = zeros(p); o.m = zeros(p); o.v = zeros(p))
function direction!(o::ADAM)
    γ = o.weight(o.n)
    s = γ * sqrt(1 - o.β2 ^ o.n) / (1 - o.β1 ^ o.n)
    for i in eachindex(o.δ)
        gi = o.δ[i]
        o.m[i] = smooth(gi,      o.m[i], o.β1)
        o.v[i] = smooth(gi * gi, o.v[i], o.β2)
        o.δ[i] = s * o.m[i] / (sqrt(o.v[i]) + ϵ)
    end
end
function Base.merge!(o::ADAM, o2::ADAM)
    o.n += o2.n 
    smooth!(o.m, o2.m, nobs(o2) / nobs(o))
    smooth!(o.v, o2.v, nobs(o2) / nobs(o))
    o
end