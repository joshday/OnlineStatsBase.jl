module OnlineStatsBase

import StatsBase

export AbstractSeries, OnlineStat, StochasticStat, Weight

abstract type Weight end
abstract type OnlineStat{INDIM, OUTDIM} end
abstract type StochasticStat{I, O} <: OnlineStat{I, O} end
abstract type AbstractSeries end

end
