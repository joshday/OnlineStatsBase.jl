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

@testset "Testing Stats" begin
#-----------------------------------------------------------------------# Counter
println("  > Counter")
@testset "Counter" begin
    o = fit!(Counter(Int), 1:10)
    @test value(o) == 10
    o2 = fit!(Counter(Int), 1)
    @test value(merge!(o, o2)) == 11
    ==(mergevals(Counter(), y, y2)...)
end

#-----------------------------------------------------------------------# CountMap
println("  > CountMap")
@testset "CountMap" begin
    a = fit!(CountMap(Bool), x)
    @test sort(value(a)) == sort(countmap(x))
    @test OnlineStatsBase.pdf(a, true) == mean(x)
    @test OnlineStatsBase.pdf(a, false) == mean(!, x)
    @test OnlineStatsBase.pdf(a, 2) == 0.0
    @test keys(a) == keys(a.value)
    @test values(a) == values(a.value)
    @test a[true] == a.value[true]
    @test OnlineStatsBase.nkeys(a) == 2

    b = fit!(CountMap(Int), z)
    @test sort(value(b)) == sort(countmap(z))
    for i in 1:11
        @test OnlineStatsBase.pdf(b, i) ≈ sum(==(i), z) / n
    end
    @test all(x -> 0 < x < 1, OnlineStatsBase.probs(b))
    @test OnlineStatsBase.probs(b, 11:13) == zeros(3)

    @test ==(mergevals(CountMap(Bool), x, x2)...)
    @test ==(mergevals(CountMap(Int), z, z2)...)
    @test ==(mergevals(CountMap{Bool}(), x, x2)...)
    @test ==(mergevals(CountMap{Int}(), z, z2)...)
    @test ==(mergevals(CountMap(Dict{Bool,Int}()), x, x2)...)
    @test ==(mergevals(CountMap(Dict{Int,Int}()), z, z2)...)
end
#-----------------------------------------------------------------------# CovMatrix
println("  > CovMatrix")
@testset "CovMatrix" begin
    o = fit!(CovMatrix(), eachrow(ymat))
    @test OnlineStatsBase.nvars(o) == size(ymat, 2)
    @test value(o) ≈ cov(ymat)
    @test cov(o) ≈ cov(ymat)
    @test cor(o) ≈ cor(ymat)
    @test all(x -> ≈(x...), zip(var(o), var(ymat; dims=1)))
    @test all(x -> ≈(x...), zip(std(o), std(ymat; dims=1)))
    @test all(x -> ≈(x...), zip(mean(o), mean(ymat; dims=1)))

    @test ≈(mergevals(CovMatrix(), OnlineStatsBase.eachrow(ymat), OnlineStatsBase.eachrow(ymat2))...)
    @test ≈(mergevals(CovMatrix(), OnlineStatsBase.eachcol(ymat'), OnlineStatsBase.eachcol(ymat2'))...)
    @test ≈(mergevals(CovMatrix(Complex{Float64}), OnlineStatsBase.eachrow(ymat * im), OnlineStatsBase.eachrow(ymat2))...)
    @test ≈(mergevals(CovMatrix(Complex{Float64}), OnlineStatsBase.eachrow(ymat * im), OnlineStatsBase.eachrow(ymat2 * im))...)
end
#-----------------------------------------------------------------------# Extrema
println("  > Extrema")
@testset "Extrema" begin
    o = fit!(Extrema(), y)
    @test extrema(o) == extrema(y)
    @test minimum(o) == minimum(y)
    @test maximum(o) == maximum(y)

    @test value(fit!(Extrema(Bool), x)) == extrema(x)
    @test value(fit!(Extrema(Int), z)) == extrema(z)

    @test ==(mergevals(Extrema(), y, y2)...)

    o = fit!(Extrema(Date), Date(2010):Day(1):Date(2011))
    @test minimum(o) == Date(2010)
    @test maximum(o) == Date(2011)

    @test value(fit!(Extrema(Char), 'a':'z')) == ('a', 'z')
    @test value(fit!(Extrema(Char), "abc")) == ('a', 'c')
    @test value(fit!(Extrema(String), ["a", "b"])) == ("a", "b")
end
#-----------------------------------------------------------------------# Group
@testset "Group" begin
    o = fit!(5Mean(), OnlineStatsBase.eachrow(ymat))
    @test o[1] == first(o)
    @test o[end] == last(o)
    @test 5Mean() == 5Mean()
    @test collect(map(value, value(o))) ≈ vec(mean(ymat, dims=1))

    o2 = Group(m1=Mean(), m2=Mean(), m3=Mean(), m4=Mean(), m5=Mean())
    fit!(o2, eachrow(ymat))
    @test collect(map(value, value(o2))) ≈ vec(mean(ymat, dims=1))
    @test length(o2) == 5

    a, b = mergevals(
        Group(Mean(), Variance(), Sum(), Moments(), Mean()),
        OnlineStatsBase.eachrow(ymat),
        OnlineStatsBase.eachrow(ymat2)
    )
    for (ai, bi) in zip(a, b)
        @test value(ai) ≈ value(bi)
    end

    @test length(values(a)) == 5

    c = fit!(Group([Mean(), Mean()]), zip(1:10, 1:10))
    for m in c
        @test value(m) ≈ mean(1:10)
    end
end
#-----------------------------------------------------------------------# GroupBy
@testset "GroupBy" begin
    @test GroupBy(Bool, Mean()) == GroupBy(Bool, Mean())
    d = value(fit!(GroupBy(Bool, Mean()), zip(x,y)))
    @test value(d[true]) ≈ mean(y[x])
    @test value(d[false]) ≈ mean(y[map(!,x)])
    string(GroupBy(Bool, Mean()))

    a, b = mergevals(GroupBy(Int, Mean()), zip(z,y), zip(z2, y2))
    for (ai,bi) in zip(values(sort(a)), values(sort(b)))
        @test value(ai) ≈ value(bi)
    end
    @test value(a[1]) ≈ value(b[1])
    string(a)
end
#-----------------------------------------------------------------------# Mean
println("  > Mean")
@testset "Mean" begin
    o = fit!(Mean(), y)
    @test value(o) ≈ mean(y)
    @test mean(o) ≈ mean(y)
    @test ≈(mergevals(Mean(), y, y2)...)
end
#-----------------------------------------------------------------------# Moments
@testset "Moments" begin
    o = fit!(Moments(), y)
    @test value(o) ≈ [mean(y), mean(y .^ 2), mean(y .^ 3), mean(y .^ 4)]
    @test mean(o) ≈ mean(y)
    @test var(o) ≈ var(y)
    @test std(o) ≈ std(y)
    @test skewness(o) ≈ skewness(y)
    @test kurtosis(o) ≈ kurtosis(y)
    for (v1,v2) in zip(mergevals(Moments(), y, y2)...)
        @test v1 ≈ v2
    end
end
#-----------------------------------------------------------------------# Series/FTSeries
println("  > Series/FTSeries")
@testset "Series/FTSeries" begin
    @testset "Series" begin
        a, b = mergevals(Series(Mean(), Variance()), y, y2)
        @test a[1] ≈ b[1]
        @test a[2] ≈ b[2]

        a, b = mergevals(Series(m=Mean(), v=Variance()), y, y2)
        @test a.m ≈ b.m
        @test a.v ≈ b.v
    end

    @testset "FTSeries" begin
        o = fit!(FTSeries(Mean(); transform=abs), y)
        @test value(o)[1] ≈ mean(abs, y)

        data = vcat(y, fill(missing, 20))
        o = fit!(FTSeries(Mean(); transform=abs, filter=!ismissing), data)
        @test value(o)[1] ≈ mean(abs, y)
        @test o.nfiltered == 20

        o2 = fit!(FTSeries(Float64, Mean(); transform=abs, filter=!ismissing), data)
        @test value(o2)[1] ≈ mean(abs, y)
        @test o2.nfiltered == 20

        a, b = mergestats(FTSeries(Mean(), Variance(); transform=abs, filter=!ismissing), y, y2)
        @test a.nfiltered == b.nfiltered
    end
end

#-----------------------------------------------------------------------# Sum
println("  > Sum")
@testset "Sum" begin
    @test sum(fit!(Sum(Int), x)) == sum(x)
    @test value(fit!(Sum(), y)) ≈ sum(y)
    @test value(fit!(Sum(Int), z)) == sum(z)

    @test ==(mergevals(Sum(Int), x, x2)...)
    @test ≈(mergevals(Sum(), y, y2)...)
    @test ==(mergevals(Sum(Int), z, z2)...)
end
#-----------------------------------------------------------------------# Variance
println("  > Variance")
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

end # end "Test Stats"