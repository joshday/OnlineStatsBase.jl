function test_merge(o1, o2, y1, y2)
    s1 = Series(y1, o1)
    s2 = Series(y2, o2)
    merge!(s1, s2)
    fit!(s2, y1)
    @test value(s1) == value(s1)
end

@testset "Exact Merge" begin
    test_merge(Mean(), Mean(), randn(10), randn(10))
    test_merge(Variance(), Variance(), randn(10), randn(10))
    test_merge(CovMatrix(4), CovMatrix(4), randn(100,4), randn(100,4))
end
