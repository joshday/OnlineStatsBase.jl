__precompile__(true)

module OnlineStatsBase

export AbstractSeries, OnlineStat,
    Weight, EqualWeight, BoundedEqualWeight, ExponentialWeight, LearningRate,
    LearningRate2, McclainWeight, HarmonicWeight

include("weight.jl")

#============================================================================= OnlineStat
`OnlineStat{I, O, W}` is an abstract type parameterized by the input and
output type/dimension `I` and `O` as well as the default weight type `W`.
The supported `I` and `O` value are:
    0       = Union{Number, Symbol, AbstractString} (ScalarOb)
    1       = AbstractVector or Tuple
    2       = AbstractMatrix
    -1      = unknown
    (1, 0)  = (x,y) pair

---
A new OnlineStat should define `StatsBase.fit!(o::MyStat, y::InputType, w::Float64)``
where `InputType` depends on `I`

---
If the OnlineStat is mergeable, it should define
- `merge!(o1::MyStat, o2::MyStat, w::Float64)`
where `w` is the influence (between 0 and 1) `o2` should have on `o1`

---
If the OnlineStat's value is not updated with fit!, it should define
`_value(o)`, which calculates the value
==============================================================================#
abstract type OnlineStat{In, Out, Weight} end

function Base.show(io::IO, o::OnlineStat)
    print(io, name(o), "(")
    showcompact(io, _value(o))
    print(io, ")")
end

Base.copy(o::OnlineStat) = deepcopy(o)
Base.map(f::Function, o::OnlineStat) = f(o)

function Base.:(==){T <: OnlineStat}(o1::T, o2::T)
    nms = fieldnames(o1)
    all(getfield.(o1, nms) .== getfield.(o2, nms))
end

Base.merge{T <: OnlineStat}(o::T, o2::T, wt::Float64) = merge!(copy(o), o2, wt)
function Base.merge!{O <: OnlineStat}(o1::O, o2::O, wt::Float64)
    error("$(typeof(o1)) is not a mergeable OnlineStat")
end

Base.start(o::OnlineStat) = false
Base.next(o::OnlineStat, state) = o, true
Base.done(o::OnlineStat, state) = state

input{INDIM}(o::OnlineStat{INDIM}) = INDIM
function input(t::Tuple)
    I = input(t[1])
    for ti in t
        input(ti) != I && throw(ArgumentError("Inputs must match. Found: $(input.(t))"))
    end
    I
end

_value(o::OnlineStat) = getfield(o, fieldnames(o)[1])

weight{I,O,W}(o::OnlineStat{I,O,W}) = W()
function weight(t::Tuple)
    w = weight(t[1])
    if !all(map(x -> weight(x) == w, t))
        throw(ArgumentError("Default weights differ.  Weight must be specified"))
    end
    w
end

#============================================================================= AbstractSeries
An AbstractSeries contains a Weight `weight` and tuple of OnlineStats `stats`,
==============================================================================#
"A container for a `Weight` and at least one `OnlineStat`"
abstract type AbstractSeries end

Base.copy(o::AbstractSeries) = deepcopy(o)

function Base.show(io::IO, s::AbstractSeries)
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
nobs(o::AbstractSeries) = nobs(o.weight)
nups(o::AbstractSeries) = nups(o.weight)
weight(o::AbstractSeries,         n2::Int = 1) = weight(o.weight, n2)
weight!(o::AbstractSeries,        n2::Int = 1) = weight!(o.weight, n2)
updatecounter!(o::AbstractSeries, n2::Int = 1) = updatecounter!(o.weight, n2)

function Base.merge{T <: AbstractSeries}(s1::T, s2::T, w::Float64)
    merge!(copy(s1), s2, w)
end
function Base.merge{T <: AbstractSeries}(s1::T, s2::T, method::Symbol = :append)
    merge!(copy(s1), s2, method)
end
function Base.merge!{T <: AbstractSeries}(s1::T, s2::T, method::Symbol = :append)
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
function Base.merge!{T <: AbstractSeries}(s1::T, s2::T, w::Float64)
    n2 = nobs(s2)
    n2 == 0 && return s1
    0 <= w <= 1 || throw(ArgumentError("weight must be between 0 and 1"))
    updatecounter!(s1, n2)
    merge!.(s1.stats, s2.stats, w)
    s1
end


#============================================================================= Show
- `show_fields` prints things like "(field1 = val1, field2 = val2)"
- `fields_to_show` tells `show_fields` which values to print
- `name` prints
    - "MyModule.MyType{T, S}" for withmodule=true, withparams=true
    - "MyModule.MyType"       for withmodule=true, withparams=false
    - "MyType"                for withmodule=false, withparams=false

Example:

If I want to print "MyModule.MyType{T, S}(field1 = val1, field2 = val2)"

function Base.show(io::IO, t::MyType)
    print(io, name(t), true, true)
    show_fields(io, t)
end
==============================================================================#
function show_fields(io::IO, o)
    nms = fields_to_show(o)
    print(io, "(")
    for nm in nms
        print(io, "$nm = $(getfield(o, nm))")
        nm != nms[end] && print(io, ", ")
    end
    print(io, ")")
end

fields_to_show(o) = fieldnames(o)

header(io::IO, s::AbstractString) = println(io, "▦ $s" )

function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        # remove text that ends in period:  OnlineStats.Mean -> Mean
        s = replace(s, r"([a-zA-Z]*\.)", "")
    end
    if !withparams
        # replace everything from "{" to the end of the string
        s = replace(s, r"\{(.*)", "")
    end
    s
end

end
