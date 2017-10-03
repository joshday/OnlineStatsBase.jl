"""
    Series(stats...)
    Series(data, stats...)
    Series(weight, stats...)
    Series(weight, data, stats...)
A Series is a container for a Weight and any number of OnlineStats.  Updating the Series
with `fit!(s, data)` will update the OnlineStats it holds according to its Weight.

### Examples
    Series(randn(100), Mean(), Variance())
    Series(ExponentialWeight(.1), Mean())

    s = Series(Mean())
    fit!(s, randn(100))
    s2 = Series(randn(123), Mean())
    merge(s, s2)
"""
struct Series{I, OS <: Union{OnlineStat, Tuple}, W <: Weight}
    weight::W
    stats::OS
end
# These act as inner constructors
Series(wt::Weight, t::Tuple)      = Series{input(t), typeof(t), typeof(wt)}(wt, t)
Series(wt::Weight, o::OnlineStat) = Series{input(o), typeof(o), typeof(wt)}(wt, o)

# empty
Series(t::Tuple)         = Series(weight(t), t)
Series(o::OnlineStat)    = Series(weight(o), o)
Series(o::OnlineStat...) = Series(weight(o), o)
Series(wt::Weight, o::OnlineStat, os::OnlineStat...) = Series(wt, tuple(o, os...))


#============================================================================= Series
An Series contains a Weight `weight` and tuple of OnlineStats `stats`,
==============================================================================#
Base.copy(o::Series) = deepcopy(o)

function Base.show(io::IO, s::Series)
    header(io, name(s, false, true))
    print(io, "┣━━ "); println(io, s.weight)
    print(io, "┗━━ Tracking")
    names = ifelse(isa(s.stats, Tuple), name.(s.stats), tuple(name(s.stats)))
    indent = maximum(length.(names))
    n = length(names)
    i = 0
    for o in s.stats
        i += 1
        char = ifelse(i == n, "┗━━", "┣━━")
        print(io, "\n    $char ", o)

    end
end

# helpers for weight
nobs(o::Series) = nobs(o.weight)
nups(o::Series) = nups(o.weight)
weight(o::Series,         n2::Int = 1) = weight(o.weight, n2)
weight!(o::Series,        n2::Int = 1) = weight!(o.weight, n2)
updatecounter!(o::Series, n2::Int = 1) = updatecounter!(o.weight, n2)

function Base.merge{T <: Series}(s1::T, s2::T, w::Float64)
    merge!(copy(s1), s2, w)
end
function Base.merge{T <: Series}(s1::T, s2::T, method::Symbol = :append)
    merge!(copy(s1), s2, method)
end
function Base.merge!{T <: Series}(s1::T, s2::T, method::Symbol = :append)
    n2 = nobs(s2)
    n2 == 0 && return s1
    updatecounter!(s1, n2)
    if method == :append
        merge!.(s1.stats, s2.stats, weight(s1, n2))
    elseif method == :mean
        merge!.(s1.stats, s2.stats, (weight(s1) + weight(s2)))
    elseif method == :singleton
        merge!.(s1.stats, s2.stats, weight(s1))
    else
        throw(ArgumentError("method must be :append, :mean, or :singleton"))
    end
    s1
end
function Base.merge!{T <: Series}(s1::T, s2::T, w::Float64)
    n2 = nobs(s2)
    n2 == 0 && return s1
    0 <= w <= 1 || throw(ArgumentError("weight must be between 0 and 1"))
    updatecounter!(s1, n2)
    merge!.(s1.stats, s2.stats, w)
    s1
end
