module OnlineStatsBaseTests
using OnlineStatsBase, Base.Test

O = OnlineStatsBase

statlist = [CStat(Mean()), CovMatrix(3), Diff(), Extrema(), HyperLogLog(10), KMeans(4,3),
            LinReg(5), Mean(), Moments(), OHistogram(-4:.1:4), OrderStats(10),
            QuantileMM(), QuantileMSPI(), QuantileSGD(), ReservoirSample(10), Sum(),
            Variance()]

#-----------------------------------------------------------------------# Printing
info("Messy output for show method coverage")
for o in statlist
    println(o)
    typeof(o) <: OnlineStat{0} && println(2o)
    println()
end
println("\n\n")

#-----------------------------------------------------------------------# Weight
@testset "Weight" begin
function test_weight(w::O.Weight, f::Function)
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

#-----------------------------------------------------------------------# Series
@testset "Series" begin 
@testset "Constructors" begin 
    y = randn(10)
    @test Series(Mean()) == Series(Mean())
    @test Series(Mean(), Variance()) == Series(Mean(), Variance())
    s = Series(Mean())
    fit!(s, y)
    @test s == Series(y, Mean())
    @test_throws Exception Series(Mean(), CovMatrix(3))
    @test Series(EqualWeight(), Mean()) == Series(EqualWeight(), Mean())
    @test_throws Exception Series(Mean(), QuantileMM())
end
@testset "methods" begin 
    @test Series(Mean()).weight == EqualWeight()
end
@testset "merging" begin 
    @test_warn "defined" merge(Series(rand(5), Diff()), Series(rand(5), Diff()))
end
end  # Series

#-----------------------------------------------------------------------# Stats 
# include("test_stats.jl")
# @testset "Test Stats" begin 
# @testset "CStat" begin 
#     o = CStat(Mean())
#     s = Series(complex(y), o)
#     @test value(o)[1] ≈ mean(y)
#     @test value(o)[2] == 0.0

#     o = CStat(Mean())
#     y_im = y * im
#     Series(y_im, o)
#     @test value(o)[1] == 0.0
#     imval = mean(map(x -> x.im, y_im))
#     @test value(o)[2] ≈ imval

#     o2 = CStat(Mean())
#     merge!(o, o2, .5) 
#     @test value(o)[2] ≈ imval / 2
# end
# @testset "CovMatrix" begin 
#     o = CovMatrix(5)
#     s = Series(x, o)
#     @test cov(o) ≈ cov(x)
#     @test cor(o) ≈ cor(x)
# end
# end  # OnlineStats


# @testset "mapblocks" begin
#     for o = [randn(6), randn(6,2), (randn(7,2), randn(7))]
#         i = 0
#         mapblocks(5, o) do x
#             i += 1
#         end
#         @test i == 2
#     end

#     # # (1, 0) input
#     # s = Series(LinReg(5))
#     x, y = randn(100,5), randn(100)
#     # mapblocks(10, (x,y)) do xy
#     #     fit!(s, xy)
#     # end
#     # s2 = Series((x,y), LinReg(5))
#     # @test nobs(s2) == nobs(s)
#     # @test s == s2

#     # s3 = Series(LinReg(5))
#     # mapblocks(11, (x', y), Cols()) do xy
#     #     fit!(s3, xy, Cols())
#     # end
#     # @test nobs(s3) == 100
#     # @test all(value(s) .≈ value(s3))

#     # 1 input
#     s4 = Series(CovMatrix(5))
#     mapblocks(11, x) do xi
#         fit!(s4, xi)
#     end
#     s5 = Series(CovMatrix(5))
#     mapblocks(11, x', Cols()) do xi
#         fit!(s5, xi, Cols())
#     end
#     @test s4 == s5

#     @test_throws Exception mapblocks(sum, 10, (x,y), Cols())
# end



# include("test_stats.jl")
end #module
