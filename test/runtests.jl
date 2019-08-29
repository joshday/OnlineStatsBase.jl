using OnlineStatsBase, LearnBase, OrderedCollections, StatsBase, Statistics, Dates, Test

import OnlineStatsBase: _fit!, _merge!, Mean, Variance

include("test_stats.jl")

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

#-----------------------------------------------------------------------# Iteration
@testset "Iteration" begin
    x, y = randn(100,10), randn(100)
    @test fit!(Counter(AbstractVector), OnlineStatsBase.eachrow(x)).n == 100
    @test fit!(Counter(AbstractVector), OnlineStatsBase.eachcol(x)).n == 10
    @test fit!(Counter(Tuple),  zip(OnlineStatsBase.eachrow(x), y)).n == 100
    @test fit!(Counter(Tuple),  zip(OnlineStatsBase.eachcol(x), y)).n == 10

    for (j, xj) in enumerate(OnlineStatsBase.eachcol(x))
        @test xj == x[:, j]
    end
    for (i, xi) in enumerate(OnlineStatsBase.eachrow(x))
        @test xi == x[i, :]
    end
    @inferred OnlineStatsBase.eachrow(x)
    @inferred OnlineStatsBase.eachcol(x)
    @inferred zip(OnlineStatsBase.eachrow(x), y)
    @inferred zip(OnlineStatsBase.eachcol(x), y)

    @test length(OnlineStatsBase.eachcol(x)) == 10
    @test length(OnlineStatsBase.eachrow(x)) == 100
    @test length(zip(OnlineStatsBase.eachrow(x), y)) == 100
end



