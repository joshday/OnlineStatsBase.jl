using Compat, Compat.Test, OnlineStatsBase, LearnBase

import OnlineStatsBase: OnlineStat, _fit!, fit!,
    EqualWeight, ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight,
    McclainWeight
#-----------------------------------------------------------------------#
mutable struct FakeStat <: OnlineStat{Number}
    n::Int 
end 
_fit!(o::FakeStat, y) = (o.n += 1)

struct FakeStat2 <: OnlineStat{Number} end 
println(FakeStat(0))

@testset "FakeStat" begin 
    o = FakeStat(0)
    @test value(o) == 0 
    @test nobs(o) == 0 
    _fit!(o, randn())
    @test value(o) == nobs(o) == 1
    @test FakeStat(10) == FakeStat(10)
    @test FakeStat(0) != FakeStat2()
    @test value(fit!(FakeStat(0), rand(100))) == 100
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
test_weight(@inferred(max(EqualWeight(), .1)),      i -> max(.1, 1 / i))
test_weight(@inferred(max(LearningRate(.6), .1)),   i -> max(.1, 1 / i^.6))
test_weight(@inferred(.1 * EqualWeight()),          i -> .1 * (1 / i))
test_weight(@inferred(.1 * EqualWeight()),          i -> .1 * (1 / i))
test_weight(max(.5 * EqualWeight(), .1),            i -> max(.1, .5 / i))

@test ExponentialWeight(20) == ExponentialWeight(2 / 21)
@test max(.1, EqualWeight()) == max(EqualWeight(), .1)
@test .1 * EqualWeight() == EqualWeight() * .1
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
end  # Weight