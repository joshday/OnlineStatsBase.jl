module OnlineStatsBaseTests
using Base.Test
import OnlineStatsBase: OnlineStat, ExactStat, StochasticStat, Weight, EqualWeight, 
    ExponentialWeight, LearningRate, LearningRate2, HarmonicWeight, McclainWeight, 
    Bounded, Scaled, _value, default_weight, name

#-----------------------------------------------------------------------# show
for w in [EqualWeight(), ExponentialWeight(), LearningRate(), LearningRate2(),
          HarmonicWeight()]
    println(w)
end
struct Thing{T} a::T end
println(name(Thing(1)))
println(name(Thing(1), true, false))

#-----------------------------------------------------------------------# Weight
@testset "Weight" begin
function test_weight(w::Weight, f::Function)
    @test w == copy(w)
    for i in 1:20
        @test w(i) == f(i)
    end
end
test_weight(@inferred(EqualWeight()),                   i -> 1 / i)
test_weight(@inferred(ExponentialWeight(.1)),           i -> ifelse(i==1, 1.0, .1))
test_weight(@inferred(LearningRate(.6)),                i -> 1 / i^.6)
test_weight(@inferred(LearningRate2(.5)),               i -> 1 / (1 + .5*(i-1)))
test_weight(@inferred(HarmonicWeight(4.)),              i -> 4 / (4 + i - 1))
test_weight(@inferred(Bounded(EqualWeight(), .1)),      i -> max(.1, 1 / i))
test_weight(@inferred(Bounded(LearningRate(.6), .1)),   i -> max(.1, 1 / i^.6))
test_weight(@inferred(Scaled(EqualWeight(), .1)),       i -> .1 * (1 / i))
test_weight(@inferred(.1 * EqualWeight()),              i -> .1 * (1 / i))

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
end  # Weight

struct FakeStat <: ExactStat{0}
    μ::Float64
end
FakeStat() = FakeStat(0.0)

struct FakeStat2 <: StochasticStat{0} end

struct FakeStat3 <: OnlineStat{0} end

@testset "OnlineStat" begin 
    println(FakeStat())
    @test _value(FakeStat()) == 0.0
    @test FakeStat() == FakeStat()
    @test_warn "defined" merge(FakeStat(), FakeStat(), .5)
    @test default_weight(FakeStat()) == EqualWeight()
    @test default_weight(FakeStat2()) == LearningRate()
    @test_throws Exception default_weight((FakeStat(), FakeStat2()))
    @test default_weight((FakeStat(), FakeStat())) == EqualWeight()
    @test_throws Exception default_weight(FakeStat3())
    @test_throws Exception _fit!(FakeStat())
end
end #module
