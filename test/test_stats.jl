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

@testset "KMeans" begin
    o = KMeans(4, 3)
    Series(Y, o)
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
end

@testset "OHistogram" begin
    o = OHistogram(-5:.1:5)
    h = fit(Histogram, y, -5:.1:5; closed = :left)
    Series(y, o)
    @test o.h == h
end

@testset "OrderStats" begin
    test_exact(OrderStats(length(y)), y, sort)
    @test issorted(first(value(Series(y, OrderStats(5)))))
end

@testset "QuantileMM" begin
    y = randn(100_000)
    o = QuantileMM()
    s = Series(y, o)
    @test all(isapprox.(value(o), quantile(y, [.25, .5, .75]); atol = .1))
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
