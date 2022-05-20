using Test, Statistics, OnlineStatsBase

y, y2 = randn(100), randn(101)
w, w2 = rand(100), randn(101)

#-----------------------------------------------------------------------------# test utils
function testmerge(o::OnlineStatsBase.OnlineStat, y1, y2, cmp = ≈)
    a, b = fit!(copy(o), y1), fit!(copy(o), y2)
    merge!(a, b)
    fit!(b, y1)
    cmp(value(a), value(b))
end

#-----------------------------------------------------------------------------# fit!
@testset "fit!" begin
    @test value(fit!(Mean(), 1)) ≈ 1
    @test value(fit!(Mean(weight=Weight.Exponential(.1)), 1)) ≈ .1
    @test value(fit!(Mean(), y)) ≈ mean(y)
    @test value(merge!(fit!(Mean(), y), fit!(Mean(), y2))) ≈ mean(vcat(y, y2))
    @test value(fit!(Mean(weight=Weight.Exponential(.1)), y)) ≈ reduce((a,b) -> OnlineStatsBase.smooth(a,b,.1), y; init=0.0)
end
#-----------------------------------------------------------------------------# Mean
@testset "Mean" begin
    @test testmerge(Mean(), y, y2)
end
