#-----------------------------------------------------------------------------# Part
struct Part{D, O} <: OnlineStat{TwoThings}
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

function Base.merge!(a::Centroid, b::Centroid, astat, bstat)
    w = nobs(bstat) / nobs(astat)
    a.center = smooth(a.center, b.center, w)
    a
end

#-----------------------------------------------------------------------------# TimeBucket
mutable struct TimeBucket{T <: Dates.TimeType}
    first::T 
    last::T
end
Base.show(io::IO, b::TimeBucket) = print(io, "TimeBucket: [$(b.first), $(b.last)]")
Base.in(x, bucket::TimeBucket) = bucket.first ≤ x ≤ bucket.last
Base.isless(a::TimeBucket, b::TimeBucket) = isless(a.first, b.first)
function Base.merge!(a::TimeBucket, b::TimeBucket)
    a.first = min(a.first, b.first)
    a.last = max(a.last, b.last)
    a
end

#-----------------------------------------------------------------------------# Interval (Part)
# const interval_types = [:left_closed, :right_closed, :closed, :open]

# struct Interval{type, T}
#     a::T
#     b::T 
#     function Interval(a, b, type = :left_closed) 
#         a < b || error("$a needs to be less than $b")
#         new{type, promote_type(typeof(a), typeof(b))}(a, b)
#     end
# end
# Base.in(x, d::Interval{:open})          = (d.a < x < d.b)
# Base.in(x, d::Interval{:closed})        = (d.a ≤ x ≤ d.b)
# Base.in(x, d::Interval{:left_closed})   = (d.a ≤ x < d.b)
# Base.in(x, d::Interval{:right_closed})  = (d.a < x ≤ d.b)
# function Base.show(io::IO, d::Interval{type}) where {type}
#     l = type in [:open, :right_closed] ? '(' : '['
#     r = type in [:open, :left_closed] ? ')' : ']'
#     print(io, "Interval: $l$(d.a), $(d.b)$r")
# end

# function _merge(a::Part{Interval{:left_closed}}, b::Part{Interval{:closed}})
#     error("not implemented yet")
# end

