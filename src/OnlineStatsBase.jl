__precompile__(true)

module OnlineStatsBase

export OnlineStat,
    Weight, EqualWeight, BoundedEqualWeight, ExponentialWeight, LearningRate,
    LearningRate2, McclainWeight, HarmonicWeight, Bounded, Scaled



const AA = AbstractArray

"""
`OnlineStat{I, O, W}` is an abstract type parameterized by the input and
output type/dimension `I` and `O` as well as the default weight type `W`.
The supported `I` and `O` value are:
    0       = Union{Number, Symbol, AbstractString} (ScalarOb)
    1       = AbstractVector or Tuple
    2       = AbstractMatrix
    -1      = unknown
    (1, 0)  = (x, y) pair of (vector, scalar)

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
"""
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

#-----------------------------------------------------------------------#
include("weight.jl")
include("series.jl")






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
