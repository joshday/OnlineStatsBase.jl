using OnlineStatsBase, OrderedCollections, StatsBase, Statistics, Dates, Test

import OnlineStatsBase: _fit!, _merge!

O = OnlineStatsBase

include("test_stats.jl")

#-----------------------------------------------------------------------# Random
@testset "Random Stuff" begin
string(Series(Mean(), Variance()))
string(Group(Series(Mean(), Variance()), Extrema()))
string(Mean())
string(McclainWeight())
string(Part(Counter(), O.Centroid(0)))
string(Part(Counter(), O.ClosedInterval(0,1)))
@test Mean() != Variance()
@test !(Mean() == Variance())
@test Mean() == merge(Mean(), Mean())
@test Mean() == merge(Mean(), fit!(Variance(), 1:5))
@test_throws Exception fit!(Mean(), "abc")
@test_throws Exception OnlineStatsBase._fit!(Mean(), "a")
@test collect(OnlineStatsBase.neighbors([1,3,5])) == [(1,3), (3,5)]
end

#-----------------------------------------------------------------------# Weight
@testset "Weight" begin
function test_weight(w::OnlineStatsBase.Weight, f::Function)
    println("  > $w")
    @test w == copy(w)
    for i in 1:20
        @test w(i) == f(i)
    end
end
test_weight(@inferred(EqualWeight()),               i -> 1 / i)
test_weight(@inferred(ExponentialWeight(.1)),       i -> ifelse(i==1, 1.0, .1))
test_weight(@inferred(LearningRate(.6)),            i -> 1 / i^.6)
test_weight(@inferred(LearningRate2(.5)),           i -> 1 / (1 + .5*(i-1)))
test_weight(@inferred(HarmonicWeight(4.)),          i -> 4 / (4 + i - 1))

@test ExponentialWeight(20) == ExponentialWeight(2 / 21)

@testset "McclainWeight" begin
    w = McclainWeight(.1)
    for i in 2:100
        @test .1 < w(i) < 1
    end
end
@testset "first weight is one" begin
    for w in [EqualWeight(), ExponentialWeight(), LearningRate(), LearningRate2(),
              HarmonicWeight(), McclainWeight()]
        @test w(1) == 1
    end
end
end #Weight




