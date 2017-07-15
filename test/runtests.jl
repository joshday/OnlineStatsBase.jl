module OnlineStatsBaseTests
using OnlineStatsBase, Base.Test
O = OnlineStatsBase


#-----------------------------------------------------------------------# Test Weight
struct FakeWeight <: Weight end
@test_throws Exception weight(FakeWeight())

@testset "Weight" begin
    function test_weight(w::Weight, f::Function)
        println(w)
        @test O.nobs(w) == 0
        @test O.nups(w) == 0
        for i in 1:10
            @test O.weight!(w) ≈ f(i)
            @test O.nobs(w) == i
            @test O.nups(w) == i
        end
        for i in 11:20
            OnlineStatsBase.updatecounter!(w)
            @test O.weight(w) ≈ f(i)
            @test O.nobs(w) == i
            @test O.nups(w) == i
        end
        @test w == copy(w)
    end

    test_weight(@inferred(EqualWeight()),           i -> 1 / i)
    test_weight(@inferred(BoundedEqualWeight(.1)),  i -> max(1 / i, .1))
    test_weight(@inferred(ExponentialWeight(.1)),   i -> .1)
    test_weight(@inferred(LearningRate(.6, .1)),    i -> max(1 / i^.6, .1))
    test_weight(@inferred(LearningRate2(.5, .1)),   i -> max(1 / (1 + .5*(i-1)), .1))
    test_weight(@inferred(HarmonicWeight(4.)),       i -> 4 / (4 + i - 1))

    @test ExponentialWeight(20) == ExponentialWeight(2 / 21)
    @test BoundedEqualWeight(20) == BoundedEqualWeight(2 / 21)

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

#-----------------------------------------------------------------------# Test OnlineStat
struct FakeStat <: OnlineStat{0, 0, EqualWeight} a::Int end
struct FakeStat2 <: OnlineStat{1, 0, LearningRate} a::Int end
@testset "OnlineStat" begin
    o = FakeStat(1)
    o2 = FakeStat2(2)
    for o in o
        println(o)
    end
    @test map(O._value, o) == O._value(o)
    @test O.weight(o) == EqualWeight()
    @test O.weight((o, o)) == EqualWeight()
    @test_throws Exception merge(o, copy(o))
    @test O.input(o) == 0
    @test O.input((o, o)) == 0

    @test_throws Exception O.weight((o, o2))
    @test_throws Exception O.input((o, o2))
    @test_throws Exception merge(o, copy(o), .5)
end

#-----------------------------------------------------------------------# Test Series
struct FakeSeries <: AbstractSeries
    weight::EqualWeight
    stats::FakeStat
end
@testset "Series" begin
    s = FakeSeries(EqualWeight(), FakeStat(0))
    println(s)
    @test O.nobs(s) == O.nobs(s.weight)
    @test O.nups(s) == O.nups(s.weight)
    @test O.weight(s) == O.weight(s.weight)
    @test O.weight!(s) == 1.0
    @test O.updatecounter!(s) == 2
end

end #module
