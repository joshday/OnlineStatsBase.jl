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
    merge!(a.domain, b.domain)
end

Base.diff(a::Part, b::Part) = diff(a.domain, b.domain)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------# Domains
#-----------------------------------------------------------------------------#
# Required methods:
#   - Base.merge!
#   - Base.in 
#   - Base.isless
#   - Base.diff

#-----------------------------------------------------------------------------# Centroid
mutable struct Centroid{T}
    center::T 
end

Base.in(x::T, c::Centroid{T}) where {T} = true 
Base.in(x, c::Centroid) = false
Base.isless(a::Centroid, b::Centroid) = isless(a.center, b.center)
Base.show(io::IO, c::Centroid) = print(io, "Centroid: $(c.center)")
Base.diff(a::Centroid, b::Centroid) = norm(a.center - b.center)

function Base.merge!(a::Part{Centroid{T}, O}, b::Part{Centroid{T}, O}) where {T <: Number, O}
    merge!(a.stat, b.stat)
    a.domain.center = smooth(a.domain.center, b.domain.center, nobs(b) / nobs(a))
    a
end
function Base.merge!(a::Part{Centroid{T}, O}, b::Part{Centroid{T}, O}) where {T, O}
    merge!(a.stat, b.stat)
    smooth!(a.domain.center, b.domain.center, nobs(b) / nobs(a))
    a
end

#-----------------------------------------------------------------------------# ClosedInterval
mutable struct ClosedInterval{T}
    first::T 
    last::T
    ClosedInterval(a::T, b::T) where {T} = a ≤ b ? new{T}(a,b) : error("Arguments must be ordered: [$a, $b]")
end
Base.show(io::IO, b::ClosedInterval) = print(io, "ClosedInterval: [$(b.first), $(b.last)]")
Base.in(x, bucket::ClosedInterval) = bucket.first ≤ x ≤ bucket.last
Base.isless(a::ClosedInterval, b::ClosedInterval) = isless(a.first, b.first)
function Base.diff(a::ClosedInterval, b::ClosedInterval) 
    a < b ? _value(b.first) - _value(a.last) : _value(a.first) - _value(b.last)
end

_value(x) = x 
_value(x::Dates.TimeType) = Dates.value(x)

function Base.merge!(a::ClosedInterval, b::ClosedInterval)
    a.first = min(a.first, b.first)
    a.last = max(a.last, b.last)
    a
end
