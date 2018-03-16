abstract type Updater end 
nobs(o::Updater) = o.n
abstract type SGUpdater <: Updater end

#-----------------------------------------------------------------------# SGD
mutable struct SGD{W} <: SGUpdater
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
mutable struct ADAGRAD{W} <: SGUpdater 
    δ::Vector{Float64}
    h::Vector{Float64}
    weight::W 
    n::Int
end
ADAGRAD(p=0; rate = LearningRate()) = ADAGRAD(zeros(p), zeros(p), rate, 0)
init!(o::ADAGRAD, p) = (o.δ = zeros(p); o.h = zeros(p))
function direction!(o::ADAGRAD)
    γ = o.weight(o.n)
    for i in eachindex(o.h)
        o.h[i] = smooth(o.h[i], o.δ[i] ^ 2, γ)
    end
    for i in eachindex(o.δ)
        o.δ[i] = γ * o.δ[i] / (o.h[i] + ϵ)
    end
end
function Base.merge!(o::ADAGRAD, o2::ADAGRAD)
    o.n += o2.n 
    smooth!(o.h, o2.h, nobs(o2) / nobs(o))
    o
end