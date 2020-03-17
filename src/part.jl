#-----------------------------------------------------------------------------# Part
struct Part{D, O<:OnlineStat} <: OnlineStat{TwoThings}
    stat::O 
    domain::D
end
value(o::Part) = (domain=o.domain, stat=o.stat)
Base.in(x, o::Part) = x ∈ o.domain
Base.isless(a::Part, b::Part) = isless(a.domain, b.domain)

nobs(o::Part) = nobs(o.stat)

function _fit!(o::Part, xy)
    first(xy) in o.domain || error("$(first(xy)) ∉ $(o.domain)")
    _fit!(o.stat, last(xy))
end

function _merge!(a::Part, b::Part)
    merge!(a.stat, b.stat)
    merge!(a.domain, b.domain, a.stat, b.stat)
end



#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------# Domains
#-----------------------------------------------------------------------------#
abstract type Domain end 

Base.merge!(a::Domain, b::Domain, astat, bstat) = merge!(a, b)

# Required methods:
#   - Base.merge!
#   - Base.in 
#   - Base.isless

#-----------------------------------------------------------------------------# Centroid
mutable struct Centroid{T}
    center::T 
end

Base.in(x::T, c::Centroid{T}) where {T} = true 
Base.in(x, c::Centroid) = false
Base.isless(a::Centroid, b::Centroid) = isless(a.center, b.center)
Base.show(io::IO, c::Centroid) = print(io, "Centroid: $(c.center)")

function Base.merge!(a::Centroid{<:Number}, b::Centroid{<:Number}, astat, bstat)
    w = nobs(bstat) / nobs(astat)
    a.center = smooth(a.center, b.center, w)
    a
end
function Base.merge!(a::Centroid, b::Centroid, astat, bstat)
    w = nobs(bstat) / nobs(astat)
    smooth!(a.center, b.center, w)
    a
end

#-----------------------------------------------------------------------------# ClosedInterval
mutable struct ClosedInterval{T}
    first::T 
    last::T
    ClosedInterval(a::T, b::T) where {T} = a < b ? new{T}(a,b) : error("Arguments must be ordered")
end
Base.show(io::IO, b::ClosedInterval) = print(io, "ClosedInterval: [$(b.first), $(b.last)]")
Base.in(x, bucket::ClosedInterval) = bucket.first ≤ x ≤ bucket.last
Base.isless(a::ClosedInterval, b::ClosedInterval) = isless(a.first, b.first)
function Base.merge!(a::ClosedInterval, b::ClosedInterval, astat, bstat)
    a.first = min(a.first, b.first)
    a.last = max(a.last, b.last)
    a
end
Base.merge!(a::ClosedInterval, b::ClosedInterval) = merge!(a, b, nothing, nothing)