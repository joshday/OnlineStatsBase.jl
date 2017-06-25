module OnlineStatsBase

export AbstractSeries, Series, OnlineStat, Weight

abstract type Weight                    end
abstract type OnlineStat{INDIM, OUTDIM} end
abstract type AbstractSeries            end

struct Series{I, OS <: Union{Tuple, OnlineStat{I}}, W <: Weight} <: AbstractSeries
    weight::W
    stats::OS
end

end
