using OnlineStatsBase, Base.Test
O = OnlineStatsBase


#-----------------------------------------------------------------------# Weight

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
        O.updatecounter!(w, 1_000_000)
        @test O.weight(w) > .1
    end
end
