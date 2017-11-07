#-----------------------------------------------------------------------# Aliases
const ScalarOb = Union{Number, AbstractString, Symbol}  # Observation has size "0"
const VectorOb = Union{AbstractVector, Tuple}           # Observation has size "1"
const Rows = ObsDim.First                               # Rows of a matrix are observations
const Cols = ObsDim.Last                                # Cols of a matrix are observations

# DataIterator can handle any of these inputs
const Data = Union{ScalarOb, VectorOb, AbstractMatrix, Tuple{AbstractMatrix, AbstractVector}}


#-----------------------------------------------------------------------# Series
"""
    Series(stats...)
    Series(weight, stats...)
    Series(data, weight, stats...; dim = Rows())
    Series(weight, data, stats...; dim = Rows())

Track any number of OnlineStats using a given weighting mechanism.

# Examples

    s = Series(ExponentialWeight(.1), Mean(), Variance(), Moments())
    fit!(s, randn(1000))
    value(s)

    o = CovMatrix(5)
    s = Series(randn(5, 1000), o; dim = Cols())
    cor(o)

    x, y = randn(1000, 10), randn(1000)
    s = Series(LinReg(10))
    fit!(s, (x, y))  # or fit!(s, x, y)
    value(s)
"""
struct Series{I, O <: OnlineStat, W <: Weight}
    weight::W
    o::O
end
Series(wt::Weight, o::OnlineStat) = Series{input_ndims(o), typeof(o), typeof(wt)}(wt, o)
Series(o::OnlineStat, wt::Weight = default_weight(o)) = Series(wt, o)


# # Init with data
# function Series(y::Data, wt::Weight, o::OnlineStat...; dim::ObsDimension = Rows())
#     s = Series(wt, o)
#     fit!(s, y, dim)
# end
# Series(wt::Weight, y::Data, o::OnlineStat; kw...) = Series(y, wt, o; kw...)
# function Series(y::Data, o::OnlineStat...; dim::ObsDimension = Rows())
#     s = Series(o...)
#     fit!(s, y, dim)
# end

# #-----------------------------------------------------------------------# Series Methods
stat(s::Series) = s.o
value(s::Series) = value(stat(s))

#---------------------------------------------------------------------------# fit helpers
# Iterate over each observation in `data` by the given dimension (rows or cols).
#
# A `DataIterator` is created by `eachob(data, series, dim)` only if `data` represents
# multiple observations to the `series`.  For example:
#       > `eachob(data::Vector, s::Series{0}, dim) --> DataIterator(data, dim)`
#       > `eachob(data::Vector, s::Series{1}, dim) --> (data,)`
struct DataIterator{D <: ObsDimension, T}
    data::T
    dim::D
end
Base.start(o::DataIterator) = 1
Base.done(o::DataIterator, i) = i > length(o)

# AbstractMatrix
Base.next(o::DataIterator{Rows, T}, i) where {T<:AbstractMatrix} = @view(o.data[i, :]), i + 1
Base.next(o::DataIterator{Cols, T}, i) where {T<:AbstractMatrix} = @view(o.data[:, i]), i + 1
Base.length(o::DataIterator{Rows, T}) where {T<:AbstractMatrix} = size(o.data, 1)
Base.length(o::DataIterator{Cols, T}) where {T<:AbstractMatrix} = size(o.data, 2)

# Tuple{AbstractMatrix, AbstractVector}
Base.next(o::DataIterator{Rows, T}, i) where {T<:Tuple} =
    (@view(o.data[1][i, :]), o.data[2][i]), i + 1
Base.next(o::DataIterator{Cols, T}, i) where {T<:Tuple} =
    (@view(o.data[1][:, i]), o.data[2][i]), i + 1
Base.length(o::DataIterator{D, T}) where {D, T<:Tuple} = length(o.data[2])

# For input == 0, any input should be iterated through element by element
eachob(y, s::Series{0}, dim) = y

# For input == 1, a vector is a single observation: a matrix should be iterated through by row/col
eachob(y::VectorOb,         s::Series{1}, dim) = (y,)
eachob(y::AbstractMatrix,   s::Series{1}, dim) = DataIterator(y, dim)

# For input == (1, 0)
eachob(y::Tuple{VectorOb, ScalarOb}, s::Series{(1,0)}, dim) = (y, )
eachob(y::Tuple{AbstractMatrix, AbstractVector}, s::Series{(1,0)}, dim) = DataIterator(y, dim)


#--------------------------------------------------------------------------------# fit!
"""
    fit!(s::Series, data)
    fit!(s::Series, data, w::StatsBase.AbstractWeights)

Update a Series with more data, optionally overriding the Weight.

# Example
    y = randn(100)
    w = rand(100)

    s = Series(Mean())
    fit!(s, y[1])          # one observation: use Series weight
    fit!(s, y[1], w[1])     # one observation: override weight
    fit!(s, y)              # multiple observations: use Series weight
    fit!(s, y, w[1])        # multiple observations: override each weight with w[1]
    fit!(s, y, Weights(w))  # multiple observations: y[i] uses weight w[i]

    x, y = randn(100, 5), randn(100)
    s = Series(LinReg(5))
    fit!(s, (x, y))  # or fit!(s, x, y)
"""
function fit!(s::Series, y::Data, dim::ObsDimension = Rows())
    for yi in eachob(y, s, dim)
        γ = weight!(s)
        fit!(s.o, yi, γ)
    end
    s
end
function fit!(s::Series, y::Data, w::Float64, dim::ObsDimension = Rows())
    data_it = eachob(y, s, dim)
    updatecounter!(s, length(data_it))
    for yi in data_it
        for o in stats(s)
            fit!(o, yi, w)
        end
    end
    s
end
function fit!(s::Series, y::Data, w::AbstractWeights, dim::ObsDimension = Rows())
    data_it = eachob(y, s, dim)
    length(w) == length(data_it) || throw(DimensionMismatch("weights don't match data length"))
    updatecounter!(s, length(data_it))
    for (yi, wi) in zip(data_it, w)
        for o in stats(s)
            fit!(o, yi, wi)
        end
    end
    s
end
fit!(s1::T, s2::T) where {T <: Series} = merge!(s1, s2)

function fit!(s::Series{(1,0)}, x::AbstractMatrix, y::AbstractVector, dim::ObsDimension = Rows())
    fit!(s, (x,y), dim)
end

fit!(o::OnlineStat, data) = (Series(data, o); o)


#-----------------------------------------------------------------------# Base
function Base.show(io::IO, s::Series)
    header(io, name(s, false))
    print(io, "┣━━ "); println(io, s.weight)
    print(io, "┗━━ "); println(io, stat(s))
    # print(io, "     ")
    # names = name.(stats(s))
    # n = length(names)
    # i = 0
    # for o in s.stats
    #     i += 1
    #     char = ifelse(i == n, "┗━━", "┣━━")
    #     print(io, "\n    $char ", o)
    # end
end

Base.copy(o::Series) = deepcopy(o)

function Base.:(==)(w1::T, w2::T) where {T <: Series}
    nms = fieldnames(w1)
    all(getfield.(w1, nms) .== getfield.(w2, nms))
end


#-----------------------------------------------------------------------# weight helpers
nobs(o::Series) = nobs(o.weight)
weight(o::Series,         n2::Int = 1) = weight(o.weight, n2)
weight!(o::Series,        n2::Int = 1) = weight!(o.weight, n2)
updatecounter!(o::Series, n2::Int = 1) = updatecounter!(o.weight, n2)

#-----------------------------------------------------------------------# merging
function Base.merge(s1::T, s2::T, w::Float64) where {T <: Series}
    merge!(copy(s1), s2, w)
end
function Base.merge(s1::T, s2::T, method::Symbol = :append) where {T <: Series}
    merge!(copy(s1), s2, method)
end
function Base.merge!(s1::T, s2::T, method::Symbol = :append) where {T <: Series}
    n2 = nobs(s2)
    n2 == 0 && return s1
    updatecounter!(s1, n2)
    if method == :append
        merge!.(s1.stats, s2.stats, weight(s1, n2))
    elseif method == :mean
        merge!.(s1.stats, s2.stats, (weight(s1) + weight(s2)))
    elseif method == :singleton
        merge!.(s1.stats, s2.stats, weight(s1))
    else
        throw(ArgumentError("method must be :append, :mean, or :singleton"))
    end
    s1
end
function Base.merge!(s1::T, s2::T, w::Float64) where {T <: Series}
    n2 = nobs(s2)
    n2 == 0 && return s1
    0 <= w <= 1 || throw(ArgumentError("weight must be between 0 and 1"))
    updatecounter!(s1, n2)
    merge!.(s1.stats, s2.stats, w)
    s1
end
