#-----------------------------------------------------------------------# Stats
n = 1000
x,  y,  z  = rand(Bool, n), randn(n), rand(1:10, n)
x2, y2, z2 = rand(Bool, n), randn(n), rand(1:10, n)
xs, ys, zs = vcat(x, x2), vcat(y, y2), vcat(z, z2)

p = 5
xmat,  ymat,  zmat  = rand(Bool, n, p), randn(n, p), rand(1:10, n, p)
xmat2, ymat2, zmat2 = rand(Bool, n, p), randn(n, p), rand(1:10, n, p)


function mergestats(a::OnlineStat, y1, y2)
    b = copy(a)
    fit!(a, y1)             # fit a on y1
    fit!(b, y2)             # fit b on y2
    merge!(a, b)            # merge b into a
    fit!(b, y1)             # fit b on y1
    @test nobs(a) == nobs(b) == length(y1) + length(y2)
    a, b
end
mergevals(o1::OnlineStat, y1, y2) = map(value, mergestats(o1, y1, y2))


#-----------------------------------------------------------------------# Mean
@testset "Mean" begin
    o = fit!(Mean(), y)
    @test value(o) ≈ mean(y)
    @test mean(o) ≈ mean(y)
    @test ≈(mergevals(Mean(), y, y2)...)
end
#-----------------------------------------------------------------------# Series
@testset "Series" begin
    a, b = mergevals(Series(Mean(), Variance()), y, y2)
    @test a[1] ≈ b[1]
    @test a[2] ≈ b[2]

    a, b = mergevals(Series(m=Mean(), v=Variance()), y, y2)
    @test a.m ≈ b.m
    @test a.v ≈ b.v
end
#-----------------------------------------------------------------------# FTSeries
@testset "FTSeries" begin
    o = fit!(FTSeries(Mean(); transform=abs), y)
    @test value(o)[1] ≈ mean(abs, y)

    data = vcat(y, fill(missing, 20))
    o = fit!(FTSeries(Mean(); transform=abs, filter=!ismissing), data)
    @test value(o)[1] ≈ mean(abs, y)
    @test o.nfiltered == 20
end
#-----------------------------------------------------------------------# Variance
@testset "Variance" begin
    o = fit!(Variance(), y)
    @test mean(o) ≈ mean(y)
    @test var(o) ≈ var(y)
    @test std(o) ≈ std(y)

    @test ≈(mergevals(Variance(), x, x2)...)
    @test ≈(mergevals(Variance(), y, y2)...)
    @test ≈(mergevals(Variance(Float32), Float32.(y), Float32.(y2))...; atol=.001)
    @test ≈(mergevals(Variance(), z, z2)...)
    # Issue 116
    @test std(Variance()) == 1
    @test std(fit!(Variance(), 1)) == 1
    @test std(fit!(Variance(), [1, 2])) == sqrt(.5)
end