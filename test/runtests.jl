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
@test Mean() != Variance()
@test !(Mean() == Variance())
@test Mean() == merge(Mean(), Mean())
@test Mean() == merge(Mean(), fit!(Variance(), 1:5))
@test_throws Exception fit!(Mean(), "abc")
@test_throws Exception OnlineStatsBase._fit!(Mean(), "a")
@test collect(OnlineStatsBase.neighbors([1,3,5])) == [(1,3), (3,5)]
@test isnan(value(fit!(Variance(), NaN)))
end

@testset "Broadcasting" begin
    o1 = Mean()
    o2 = Variance()
    @test tuple.(o1, o2, [1, 2]) == [(o1, o2, 1), (o1, o2, 2)]
    @test tuple.(o1, o2, [1, 2])[1][1] === o1
    @test tuple.(o1, o2, [1, 2])[2][1] === o1
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
test_weight(@inferred(ExponentialWeight(.1)),       i -> .1)
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
    for w in [EqualWeight(), LearningRate(), LearningRate2(), HarmonicWeight(), McclainWeight()]
        @test w(1) == 1
    end
end
end #Weight

@testset "Merging OnlineStats with fit!" begin
    s = 100
    x = rand(Int32, s).%100
    ranges = Iterators.partition(1:s, 10)

    v = [reduce(fit!, x[r], init=Mean()) for r in ranges]
    r1 = reduce(fit!, v, init=Mean())
    r2 = fit!(Mean(), x)

    @test isapprox(value(r1), value(r2))
    @test nobs(r1) == nobs(r2)
end
