module OnlineStatsBase

import StatsBase

export AbstractSeries, OnlineStat, StochasticStat, Weight, Series

abstract type Weight end
abstract type OnlineStat{INDIM, OUTDIM} end
abstract type StochasticStat{I, O} <: OnlineStat{I, O} end


#-----------------------------------------------------------------------# Series
abstract type AbstractSeries end

struct Series{I, OS <: Union{Tuple, OnlineStat{I}}, W <: Weight} <: AbstractSeries
    weight::W
    stats::OS
end

end
