#-----------------------------------------------------------------------------# StatWrapper
abstract type StatWrapper{T} <: OnlineStat{T} end 
nobs(o::StatWrapper) = nobs(o.stat)
value(o::StatWrapper) = value(o.stat)
_merge!(a::StatWrapper{T}, b::StatWrapper{T}) where {T} = _merge!(a.stat, b.stat)
name(o::T, args...) where {T<:StatWrapper} = name(typeof(o), args...) * "($(name(o.stat, args...)))"

#-----------------------------------------------------------------------------# CountMissing
"""
    CountMissing(stat)

Calculate a `stat` along with the count of `missing` values.  

# Example 

    o = CountMissing(Mean())
    fit!(o, [1, missing, 3])
"""
mutable struct CountMissing{T, O<:OnlineStat{T}} <: StatWrapper{Union{Missing,T}}
    stat::O
    nmissing::Int
end
CountMissing(stat::OnlineStat) = CountMissing(stat, 0)
value(o::CountMissing) = (nmissing=o.nmissing, stat=o.stat)
nobs(o::CountMissing) = nobs(o.stat) + o.nmissing

_fit!(o::CountMissing, x) = _fit!(o.stat, x)
_fit!(o::CountMissing, ::Missing) = (o.nmissing += 1)

_merge!(a::CountMissing, b::CountMissing) = (merge!(a.stat, b.stat); a.nmissing += b.nmissing)

#-----------------------------------------------------------------------------# FilterTransform
"""
    FilterTransform(stat::OnlineStat{S}, T = S; filter = x->true, transform = identity)
    FilterTransform(T => filter => transform => stat)

Wrapper around an OnlineStat that the filters and transforms its input.  Note that, depending on 
your transformation, you may need to specify the type of a single observation (`T`).

# Examples

    o = FilterTransform(Mean(), Union{Missing,Number}, filter=!ismissing)
    fit!(o, [1, missing, 3])

    o = FilterTransform(String => (x->true) => (x->parse(Int,x)) => Mean())
    fit!(o, "1")
"""
struct FilterTransform{S, T, O<:OnlineStat{T},F,F2} <: StatWrapper{S}
    stat::O
    filter::F 
    transform::F2
end
FilterTransform(intype::DataType, stat::OnlineStat; kw...) = FilterTransform(stat, intype; kw...)
function FilterTransform(stat::OnlineStat{T}, intype=T; filter=always_true, transform=identity) where {T}
    FilterTransform{intype, T, typeof(stat), typeof(filter), typeof(transform)}(stat, filter, transform)
end
function FilterTransform(p::Pair{DataType, <:Pair{<:Function, <:Pair{<:Function, <:OnlineStat}}})
    FilterTransform(p[1], p[2][2][2]; filter=p[2][1], transform=p[2][2][1])
end

_fit!(o::FilterTransform, y) = o.filter(y) && _fit!(o.stat, o.transform(y))

additional_info(o::FilterTransform) = (; filter=o.filter, transform=o.transform)

always_true(x) = true


#-----------------------------------------------------------------------------# SkipMissing 
"""
    SkipMissing(stat)

Wrapper around an OnlineStat that will skip over `missing` values.

# Example 

    o = SkipMissing(Mean())

    fit!(o, [1, missing, 3])
"""
struct SkipMissing{T, O<:OnlineStat{T}} <: StatWrapper{Union{Missing,T}}
    stat::O 
    SkipMissing(stat::OnlineStat{T}) where {T} = new{T, typeof(stat)}(stat)
end
_fit!(o::SkipMissing, x::Missing) = nothing 
_fit!(o::SkipMissing, x) = _fit!(o.stat, x)
Base.skipmissing(o::OnlineStat) = SkipMissing(o)

#-----------------------------------------------------------------------------# TryCatch
"""
    TryCatch(stat; error_limit=1000, error_message_limit=90)

Wrap each call to `fit!` in a `try`-`catch` block and track the errors encountered (via [`CountMap`](@ref)).  Errors will stop
being tracked after `error_limit` unique errors are encountered.  Only the first `error_message_limit`
characters of each error message will be recorded.

# Example 

    o = TryCatch(Mean())

    fit!(o, [1, missing, 3])

    OnlineStatsBase.errors(o)
"""
struct TryCatch{T, O<:OnlineStat{T}} <: StatWrapper{T}
    stat::O
    errors::CountMap{String}
    error_limit::Int
    error_message_limit::Int
end
function TryCatch(stat::OnlineStat; error_limit=1000, error_message_limit=90) 
    TryCatch(stat, CountMap(String), error_limit, error_message_limit)
end

errors(o::TryCatch) = value(o.errors)

function additional_info(o::TryCatch)
    ex = errors(o)
    nex = length(ex)
    msg = length(ex) ≥ o.error_limit ? "$nex (limit reached)" : nex
    nex == 0 ? () : (; errors=msg)
end

function handle_error!(o::TryCatch, ex) 
    io = IOBuffer()
    Base.showerror(io, ex)
    s = String(take!(io))
    lim = o.error_message_limit
    s = length(s) > lim ? s[1:lim] * "..." : s
    length(value(o.errors)) < o.error_limit && _fit!(o.errors, s)
end

function fit!(o::TryCatch{T}, y::T) where {T}
    try
        _fit!(o.stat, y)
    catch ex
        handle_error!(o, ex)
    end
    o
end

function fit!(o::TryCatch{I}, y::T) where {I, T}
    try
        T == eltype(y) && error("The input for $(name(o,false,false)) is $I.  Found $T.")
        for yi in y 
            fit!(o, yi)
        end
    catch ex
        handle_error!(o, ex)
    end
    o
end
