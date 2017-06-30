module OnlineStatsBase

import StatsBase: nobs

export AbstractSeries, Series, OnlineStat, StochasticStat, Weight

#============================================================================= Weight
Subtypes of Weight need at least the fields
- `nobs`
- `nups`
and a method for
- `weight(w, n2::Int = 1)`
==============================================================================#
abstract type Weight end
# show
Base.show(io::IO, w::Weight) = (print(io, name(w)); show_fields(io, w))
fields_to_show(w::Weight) = setdiff(fieldnames(w), [:nups])
# ==
function Base.:(==){T <: Weight}(w1::T, w2::T)
    nms = fieldnames(w1)
    all(getfield.(w1, nms) .== getfield.(w2, nms))
end
# interface
nobs(w::Weight) = w.nobs
nups(w::Weight) = w.nups
function updatecounter!(w::Weight, n2::Int = 1)
    w.nobs += n2
    w.nups += 1
end
function weight!(w::Weight, n2::Int = 1)
    updatecounter!(w, n2)
    weight(w, n2)
end
function weight(w::Weight) end

#============================================================================= OnlineStat
`OnlineStat{I, O}` is an abstract type parameterized by the input and
output type/dimension `I` and `O`.  The supported `I` and `O` value are:
- 0 = Union{Number, Symbol, AbstractString}
- 1 = AbstractVector or Tuple
- 2 = AbstractMatrix
- -1 = unknown
- Distributions.Distribution

A new OnlineStat (`<: OnlineStat{I, O)}`) should define :
- StatsBase.fit!(o::MyStat, y::InputType, w::Float64)

where `InputType` depends on `I`
==============================================================================#
abstract type OnlineStat{I, O} end
abstract type StochasticStat{I, O} <: OnlineStat{I, O} end

Base.show(io::IO, o::OnlineStat) = (print(io, name(o)); show_fields(io, o))

Base.copy(o::OnlineStat) = deepcopy(o)
Base.map(f::Function, o::OnlineStat) = f(o)
Base.merge{T <: OnlineStat}(o::T, o2::T, wt::Float64) = merge!(copy(o), o2, wt)

Base.start(o::OnlineStat) = false
Base.next(o::OnlineStat, state) = o, true
Base.done(o::OnlineStat, state) = state

value(o::OnlineStat) = getfield(o, fieldnames(o)[1])

#============================================================================= AbstractSeries
An AbstractSeries contains a Weight `weight` and tuple of OnlineStats `stats`,

==============================================================================#
abstract type AbstractSeries end
Base.copy(o::AbstractSeries) = deepcopy(o)

# helpers for weight
nobs(o::AbstractSeries) = nobs(o.weight)
nups(o::AbstractSeries) = nups(o.weight)
weight(o::AbstractSeries, n2::Int = 1) = weight(o.weight, n2)
weight!(o::AbstractSeries, n2::Int = 1) = weight!(o.weight, n2)
updatecounter!(o::AbstractSeries, n2::Int = 1) = updatecounter!(o.weight, n2)


#============================================================================= Show help
- show_fields
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

header(io::IO, s::AbstractString) = print(io, "▦ $s" )

function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        s = replace(s, r"(.*)\.", "")
    end
    if !withparams
        s = replace(s, r"\{(.*)", "")
    end
    s
end

end
