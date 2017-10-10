function test_merge(o1, o2, y1, y2)
    s1 = Series(y1, o1)
    s2 = Series(y2, o2)
    merge!(s1, s2)
    fit!(s2, y1)
    @test value(s1) == value(s1)
end

function test_exact(o, y, f)
    s = Series(y, o)
    @test value(o) ≈ f(y)
end

@testset "Mean" begin
    test_exact(Mean(), randn(100), mean)
    test_merge(Mean(), Mean(), randn(100), randn(100))
end

@testset "Variance" begin
    test_exact(Variance(), randn(100), var)
    test_merge(Variance(), Variance(), randn(10), randn(10))
end

@testset "CovMatrix" begin
    test_exact(CovMatrix(4), randn(100, 4), cov)
    test_merge(CovMatrix(4), CovMatrix(4), randn(100,4), randn(100,4))
end
