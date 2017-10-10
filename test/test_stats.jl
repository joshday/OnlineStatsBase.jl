#-----------------------------------------------------------------------# helpers
function test_merge(o1, o2, y1, y2)
    s1 = Series(y1, o1)
    s2 = Series(y2, o2)
    merge!(s1, s2)
    fit!(s2, y1)
    @test value(o1) == value(o1)
end

function test_exact(o, y, f; kw...)
    s = Series(y, o; kw...)
    @test all(value(o) .≈ f(y))
end

#-----------------------------------------------------------------------# Stats
@testset "CovMatrix" begin
    Series(randn(5), CovMatrix(5))
    test_exact(CovMatrix(4), randn(100, 4), cov)
    test_exact(CovMatrix(4), randn(4, 100), x -> cov(x, 2); dim = Cols())
    test_merge(CovMatrix(4), CovMatrix(4), randn(100,4), randn(100,4))
end

@testset "Extrema" begin
    test_exact(Extrema(), randn(100), extrema)
    test_merge(Extrema(), Extrema(), randn(100), randn(100))
end

@testset "Mean" begin
    test_exact(Mean(), randn(100), mean)
    test_exact(Mean(), 1.0, mean)
    test_merge(Mean(), Mean(), randn(100), randn(100))
end

@testset "Moments" begin
    moments(x) = [mean(x), mean(x .^ 2), mean(x .^ 3), mean(x .^ 4)]
    test_exact(Moments(), randn(100), moments)
    test_merge(Moments(), Moments(), randn(100), randn(100))
end

@testset "Sum" begin
    test_exact(Sum(), rand(100), sum)
    test_merge(Sum(), Sum(), randn(100), randn(100))
end

@testset "Variance" begin
    test_exact(Variance(), randn(100), var)
    test_merge(Variance(), Variance(), randn(10), randn(10))
end
