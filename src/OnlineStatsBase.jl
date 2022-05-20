module OnlineStatsBase

using Statistics
using AbstractTrees
import StatsAPI: fit!, nobs

export Weight,
    Maximum, Mean, Minimum, Nobs, Variance,
    value, fit!

#-----------------------------------------------------------------------------# utils
name(T::Type; params = false) = replace(string(T), r"([a-zA-Z]*\.)" => "", params ? "" => "" : r"\{(.*)" => "")
name(o; params=false) = name(typeof(o); params)

#-----------------------------------------------------------------------------# weight
include("weight.jl")

#-----------------------------------------------------------------------------# OnlineStat
abstract type OnlineStat{T} end
abstract type ExactStat{T} <: OnlineStat{T} end
abstract type ApproxStat{T} <: OnlineStat{T} end

abstract type StatCollection{T} <: OnlineStat{T} end
Base.show(io::IO, o::StatCollection) = AbstractTrees.print_tree(io, o)
AbstractTrees.printnode(io::IO, o::StatCollection) = print(io, name(o, params=false))
AbstractTrees.children(o::StatCollection) = collect(o.stats)

typeof_observation(o::OnlineStat{T}) where {T} = T

(o::OnlineStat)(data) = fit!(o, data)

Broadcast.broadcastable(o::OnlineStat) = Ref(o)

value(o::T) where {T<:OnlineStat} = getfield(o, first(fieldnames(T)))

Base.:(==)(o::OnlineStat, o2::OnlineStat) = false
function Base.:(==)(o1::T, o2::T) where {T<:OnlineStat}
    nms = fieldnames(typeof(o1))
    all(getfield.(Ref(o1), nms) .== getfield.(Ref(o2), nms))
end

Base.copy(o::OnlineStat) = deepcopy(o)

nobs(o::T) where {T<:OnlineStat} = :n ∈ fieldnames(T) ? o.n : nothing

Base.empty!(o::OnlineStat) = error("$(typeof(o)) has no `Base.empty!` method.")

#-----------------------------------------------------------------------------# merge!
Base.merge!(a::OnlineStat, b::OnlineStat) = (_merge!(a, b); return a)
_merge!(a, b) = error("Merging $(name(b)) into $(name(a)) is not defined.")
Base.merge(a::OnlineStat, b::OnlineStat) = merge!(copy(a), b)

#-----------------------------------------------------------------------------# fit!
fit!(o::OnlineStat{T}, yi::T) where {T} = (_fit!(o, yi); return o)

fit!(o::OnlineStat, o2::OnlineStat) = merge!(o, o2)

function fit!(o::OnlineStat{I}, y::T) where {I,T}
    T == eltype(y) && error("The input for $(name(o)) is $I.  Found $T.")
    for yi in y
        fit!(o, yi)
    end
    o
end


#-----------------------------------------------------------------------------# smooth
smooth(a, b, γ) = a + γ * (b - a)
function smooth!(a, b, γ)
    for (i, bi) in zip(eachindex(a), b)
        @inbounds a[i] = smooth(a[i], bi, γ)
    end
end
# symmetric rank 1 update of upper triangle
function smooth_syr!(A::AbstractMatrix, x, γ)
    for j in axes(A, 2), i in 1:j
        A[i, j] = smooth(A[i, j], x[i] * conj(x[j]), γ)
    end
end

#-----------------------------------------------------------------------# Base.show
function Base.show(io::IO, o::OnlineStat)
    print(io, name(o, params=showparams(o)), ": value=", format(value(o)))
    for (k,v) in pairs(keyvalues(o))
        printstyled(io, " | ", k, '=', format(v), color=:light_black)
    end
end

showparams(o::OnlineStat) = false  # whether to show type parameters

keyvalues(o) = ()  # key->value pairs to print e.g. Mean: n=0 | value=0.0 | key=value

function format(x::Integer)
    n = string(x)
    join([n[max(end_index - 3 + 1, 1):end_index] for end_index in reverse(length(n):-3:1)], '_')  # adapted from Humanize.jl
end
format(x) = x

#-----------------------------------------------------------------------------# utils
bessel(n) = n / (n - 1)

Statistics.std(o::OnlineStat; kw...) = sqrt.(var(o; kw...))

const TwoThings{T,S} = Union{Tuple{T,S}, Pair{<:T,<:S}, NamedTuple{names, Tuple{T,S}}} where names

neighbors(x) = @inbounds ((x[i], x[i+1]) for i in eachindex(x)[1:end-1])

#-----------------------------------------------------------------------------# stats
include("stats.jl")

end #module
