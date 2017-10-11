module OnlineStatsBaseTests
using OnlineStatsBase, Base.Test, StatsBase
O = OnlineStatsBase


#-----------------------------------------------------------------------# Ugly output
@testset "Series" begin
    show(Series(Mean()))
    show(Series(Mean(), Variance()))
    println()
    Series(Mean())
    Series(LearningRate(), Mean())
    Series(randn(100), Mean())
    Series(randn(100), LearningRate(), Mean())
    Series(LearningRate(), randn(100), Mean())
    @test Series(Mean()) == Series(Mean())
    s = Series(Mean())
    fit!(s, .1, .1)
    fit!(s, randn(10), rand(10))
    s2 = copy(s)
    @test s == s2
    @test all(stats(s) .== s.stats)
    @test all(value(s) .== value.(s.stats))
    merge(s, s2, :singleton)
    merge(s, s2, :mean)
    merge(s, s2, .5)
    merge!(s, s2, .5)
    @test_throws Exception merge(s, s2, :bad_arg)
end

#-----------------------------------------------------------------------# Test Weight
@testset "Weight" begin
    function test_weight(w::Weight, f::Function)
        println(w)
        @test nobs(w) == 0
        for i in 1:10
            @test O.weight!(w) ≈ f(i)
            @test nobs(w) == i
        end
        for i in 11:20
            OnlineStatsBase.updatecounter!(w)
            @test O.weight(w) ≈ f(i)
            @test nobs(w) == i
        end
        @test w == copy(w)
    end

    test_weight(@inferred(EqualWeight()),                   i -> 1 / i)
    test_weight(@inferred(ExponentialWeight(.1)),           i -> .1)
    test_weight(@inferred(LearningRate(.6)),                i -> 1 / i^.6)
    test_weight(@inferred(LearningRate2(.5)),               i -> 1 / (1 + .5*(i-1)))
    test_weight(@inferred(HarmonicWeight(4.)),              i -> 4 / (4 + i - 1))
    test_weight(@inferred(Bounded(EqualWeight(), .1)),      i -> max(.1, 1 / i))
    test_weight(@inferred(Bounded(LearningRate(.6), .1)),   i -> max(.1, 1 / i^.6))
    test_weight(@inferred(Scaled(EqualWeight(), .1)),       i -> .1 / i)
    test_weight(@inferred(.1 * EqualWeight()),              i -> .1 / i)

    @test ExponentialWeight(20) == ExponentialWeight(2 / 21)

    @testset "McclainWeight" begin
        w = @inferred McclainWeight(.1)
        println(w)
        for j in 1:10000
            O.updatecounter!(w)
        end
        @test .1 < O.weight(w) < 1.0
        @test_throws ArgumentError McclainWeight(-1.)
        @test_throws ArgumentError McclainWeight(1.1)
    end
end

@testset "Series" begin
    Series(Mean())
    Series(Mean(), Variance())
    Series(randn(100), Mean(), Variance())
    Series(randn(100, 4), CovMatrix(4))
    @test_throws Exception Series(Mean(), QuantileMM())
end

include("test_stats.jl")
end #module
