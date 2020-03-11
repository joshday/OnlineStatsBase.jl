#-----------------------------------------------------------------------------# Part
"""
    Part(stat, domain)

A `stat` calculated over a cross section (`domain`) of another variable.

# Example 


"""
mutable struct Part{D, O<:OnlineStat} <: OnlineStat{TwoThings}
    stat::O 
    domain::D
end
value(o::Part) = (domain=o.domain, stat=o.stat)
_merge!(a::Part, b::Part) = (merge!(a.stat, b.stat); merge!(a.domain, b.domain))
Base.in(x, o::Part) = x ∈ o.domain
function _fit!(o::Part, xy)
    x, y = xy 
    x in o.domain || error("$x ∉ $(o.domain)")
    _fit!(o.stat, y)
end

#-----------------------------------------------------------------------------# Interval (Part)
const interval_types = [:left_closed, :right_closed, :closed, :open]

mutable struct Interval{type, T}
    a::T
    b::T 
    function Interval(a, b, type = :left_closed) 
        a < b || error("$a needs to be less than $b")
        new{type, promote_type(typeof(a), typeof(b))}(a, b)
    end
end
Base.in(x, d::Interval{:open})          = (d.a < x < d.b)
Base.in(x, d::Interval{:closed})        = (d.a ≤ x ≤ d.b)
Base.in(x, d::Interval{:left_closed})   = (d.a ≤ x < d.b)
Base.in(x, d::Interval{:right_closed})  = (d.a < x ≤ d.b)
function Base.show(io::IO, d::Interval{type}) where {type}
    l = type in [:open, :right_closed] ? '(' : '['
    r = type in [:open, :left_closed] ? ')' : ']'
    print(io, "Interval: $l$(d.a), $(d.b)$r")
end
merge!(i::Interval, j::Interval) = (i.a = min(i.a, j.a); i.b = min(i.b, j.b))

#-----------------------------------------------------------------------------# Centroid
mutable struct Centroid{T}
    center::T 
    n::Int
end
Base.in(x::T, c::Centroid{T}) where {T} = true 
Base.in(x, c::Centroid) = false
Base.show(io::IO, c::Centroid) = "Centroid: $(c.center)"
function Base.merge!(a::Centroid{T}, b::Centroid{T}) where {T <: Number} 
    a.n += b.n
    a.center = smooth(a.center, b.center, b.n / a.n)
end
function Base.merge!(a::Centroid, b::Centroid)
    a.n += b.n
    smooth!(a.n, b.n, b.n / a.n)
end