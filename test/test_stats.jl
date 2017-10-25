#-----------------------------------------------------------------------# helpers
# test: merge is same as fit!
function test_merge(o1, o2, y1, y2)
    s1 = @inferred Series(y1, o1)
    s2 = Series(y2, o2)
    merge!(s1, s2)
    fit!(s2, y1)
    @test value(o1) == value(o1)
end

# test: value(o) == f(y)
function test_exact(o, y, f; kw...)
    s = @inferred Series(y, o; kw...)
    @test all(value(o) .≈ f(y))
end

# test: fo(o) == fy(y)
function test_function(o, y, fo, fy; atol = 1e-10)
    @inferred Series(y, o)
    @test all(isapprox.(fo(o), fy(y), atol = atol))
end

y = randn(100)
y2 = randn(100)
Y = randn(100, 4)
Y2 = randn(100, 4)

#-----------------------------------------------------------------------# Stats
@testset "CStat" begin
    o = CStat(Mean())
    s = Series(complex(y), o)
    @test value(o)[1] ≈ mean(y)
    @test value(o)[2] == 0.0

    o = CStat(Mean())
    y_im = y * im
    Series(y_im, o)
    @test value(o)[1] == 0.0
    imval = mean(map(x -> x.im, y_im))
    @test value(o)[2] ≈ imval

    o2 = CStat(Mean())
    merge!(o, o2, .5) 
    @test value(o)[2] ≈ imval / 2
end
@testset "CovMatrix" begin
    Series(randn(5), CovMatrix(5))
    test_exact(CovMatrix(4), randn(100, 4), cov)
    test_exact(CovMatrix(4), randn(4, 100), x -> cov(x, 2); dim = Cols())
    test_merge(CovMatrix(4), CovMatrix(4), randn(100,4), randn(100,4))
    test_function(CovMatrix(4), Y, cov, cov)
    test_function(CovMatrix(4), Y, cor, cor)

    o = CovMatrix(4)
    Series(Y, o)
    for f in [mean, var, std]
        @test f(o) ≈ vec(f(Y, 1))
    end
    @test length(o) == 4
end

@testset "Diff" begin
    test_exact(Diff(Int), [1, 2], x -> diff(x)[1])
    test_exact(Diff(), [1.0, 2.0], x -> diff(x)[1])
    o = Diff()
    Series(ones(5), o)
    @test last(o) == 1.0
    @test diff(o) == 0.0
end

@testset "Extrema" begin
    test_exact(Extrema(), randn(100), extrema)
    test_merge(Extrema(), Extrema(), randn(100), randn(100))
    test_function(Extrema(), y, extrema, extrema)
end

@testset "HyperLogLog" begin
    o = HyperLogLog(10)
    @test value(o) == 0.0
    s = Series(rand(1:5, 500), o)
    @test value(o) ≈ 5 atol=.5
    Series(randn(1000), HyperLogLog(4))

    o2 = HyperLogLog(11)
    @test_throws Exception merge(o, o2)
    o3 = HyperLogLog(10)
    s2 = Series(rand(6:10, 500), o3)
    merge!(s, s2)
    @test value(o) ≈ 10 atol=.5
end

@testset "KMeans" begin
    o = KMeans(4, 3)
    Series(Y, o)
end

@testset "LinReg" begin
    n, p = 100, 10
    x = randn(n, p)
    y = x * linspace(-1, 1, p) + randn(n)
    o = LinReg(p)
    Series((x,y), o)
    @test value(o) ≈ x \ y
    @test predict(o, x, Rows()) == x * o.β
    @test predict(o, x', Cols()) ≈ predict(o, x)
    @test nobs(o) == n

    Series((randn(10), randn()), LinReg(10))

    @testset "Column obs" begin
        n, p = 100, 10
        x = randn(n, p)
        y = x * linspace(-1, 1, p) + randn(n)
        o = LinReg(p)
        Series((x', y), o; dim = Cols())
        @test value(o) ≈ x \ y
        @test predict(o, x, Rows()) == x * o.β
        @test predict(o, x', Cols()) ≈ predict(o, x)
        @test nobs(o) == n
    end

    @testset "merge" begin
        o = LinReg(5)
        o2 = LinReg(5, rand(5))
        @test_throws Exception merge!(o, o2, .5)

        o2 = LinReg(5)
        x, y = randn(100,5), randn(100)
        Series((x,y), o, o2)
        merge!(o, o2, .5)
        @test value(o) == value(o2)
    end
end

@testset "Mean" begin
    test_exact(Mean(), randn(100), mean)
    test_exact(Mean(), 1.0, mean)
    test_merge(Mean(), Mean(), randn(100), randn(100))
    test_function(Mean(), y, mean, mean)
    @test merge(Mean(), Mean(), .5) == Mean()
end

@testset "Moments" begin
    moments(x) = [mean(x), mean(x .^ 2), mean(x .^ 3), mean(x .^ 4)]
    test_exact(Moments(), randn(100), moments)
    test_merge(Moments(), Moments(), randn(100), randn(100))
    test_function(Moments(), y, mean, mean)
    test_function(Moments(), y, var, var)
    test_function(Moments(), y, std, std)
    test_function(Moments(), y, skewness, skewness; atol=.1)
    test_function(Moments(), y, kurtosis, kurtosis; atol=.1)
end

@testset "MV" begin
    test_exact(MV(4, Mean()), Y, x -> vec(mean(x, 1)))
    test_exact(MV(4, Variance()), Y, x -> vec(var(x, 1)))
    test_merge(MV(4, Mean()), MV(4, Mean()), Y, Y2)

    o = MV(4, Mean())
    Series(Y', o; dim = Cols())
    @test value(o) ≈ vec(mean(Y, 1))
    @test length(o) == 4
end

@testset "OHistogram" begin
    test_merge(OHistogram(-5:.1:5), OHistogram(-5:.1:5), y, y2)
    o = OHistogram(-5:.1:5)
    h = fit(Histogram, y, -5:.1:5; closed = :left)
    Series(y, o)
    @test o.h == h
end

@testset "OrderStats" begin
    test_merge(OrderStats(5), OrderStats(5), y, y2)
    test_exact(OrderStats(length(y)), y, sort)
    @test issorted(first(value(Series(y, OrderStats(5)))))
end

@testset "Quantile Types" begin
    y = randn(100_000)
    o = QuantileMM()
    o2 = QuantileSGD()
    o3 = QuantileMSPI()
    s = Series(y, o, o2, o3)
    @test all(isapprox.(value(o), quantile(y, [.25, .5, .75]); atol = .1))
    @test all(isapprox.(value(o2), quantile(y, [.25, .5, .75]); atol = .1))
    @test all(isapprox.(value(o3), quantile(y, [.25, .5, .75]); atol = .1))

    s2 = Series(y2, QuantileMM(), QuantileSGD(), QuantileMSPI())
    merge!(s, s2)
end


@testset "ReservoirSample" begin
    test_exact(ReservoirSample(length(y)), y, identity)
    o = ReservoirSample(10)
    Series(y, o)
    for j in value(o)
        @test j in y
    end
end

@testset "Sum" begin
    test_exact(Sum(Int), collect(1:100), sum)
    test_exact(Sum(), rand(100), sum)
    test_merge(Sum(), Sum(), randn(100), randn(100))
    test_function(Sum(), y, sum, sum)
end

@testset "Variance" begin
    test_exact(Variance(), randn(100), var)
    test_merge(Variance(), Variance(), randn(10), randn(10))
    test_function(Variance(), y, mean, mean)
    test_function(Variance(), y, var, var)
    test_function(Variance(), y, std, std)
    test_function(Variance(), randn(10), nobs, x -> 10)
end
