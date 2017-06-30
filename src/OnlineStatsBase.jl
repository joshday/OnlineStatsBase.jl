module OnlineStatsBase

export AbstractSeries, Series, OnlineStat, Weight

#-----------------------------------------------------------------------# Weight
abstract type Weight end

#-----------------------------------------------------------------------# OnlineStat
"""
`OnlineStat{I, O}` is an abstract type parameterized by the input and output
type/dimension `I`.


"""
abstract type OnlineStat{INDIM, OUTDIM} end

Base.copy(o::OnlineStat) = deepcopy(o)
Base.map(f::Function, o::OnlineStat) = f(o)
Base.start(o::OnlineStat) = false
Base.next(o::OnlineStat, state) = o, true
Base.done(o::OnlineStat, state) = state

#-----------------------------------------------------------------------# Series
abstract type AbstractSeries end

struct Series{I, OS <: Union{Tuple, OnlineStat{I}}, W <: Weight} <: AbstractSeries
    weight::W
    stats::OS
end

end
