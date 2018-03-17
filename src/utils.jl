#-----------------------------------------------------------------------# BiasVec 
"""
    BiasVec(x)

Lightweight wrapper of a vector which adds a "bias" term at the end.

# Example

    BiasVec(rand(5), 10)
"""
struct BiasVec{T, A <: VectorOb} <: AbstractVector{T}
    x::A
    bias::T
end
# typeof(x[1]) instead of eltype -> allow tuples

BiasVec(x::AbstractVector{T}) where {T} = BiasVec(x, one(T))
BiasVec(x::Tup) = BiasVec(x, one(typeof(first(x))))

Base.length(v::BiasVec) = length(v.x) + 1
Base.size(v::BiasVec) = (length(v), )
Base.getindex(v::BiasVec, i::Int) = i > length(v.x) ? v.bias : v.x[i]
Base.IndexStyle(::Type{<:BiasVec}) = IndexLinear()

#-----------------------------------------------------------------------# eachrow
struct RowsOf{T, M <: AbstractMatrix{T}}
    mat::M
    buffer::Vector{T}
end
function RowsOf(x::M) where {T, M<:AbstractMatrix{T}}
    RowsOf{T, M}(x, zeros(T, size(x, 2)))
end
eachrow(x::AbstractMatrix) = RowsOf(x)
Base.start(o::RowsOf) = 1
function Base.next(o::RowsOf, i) 
    for j in eachindex(o.buffer)
        o.buffer[j] = o.mat[i, j]
    end
    o.buffer, i + 1
end
Base.done(o::RowsOf, i) = i > size(o.mat, 1)

#-----------------------------------------------------------------------# eachcol
struct ColsOf{T, M <: AbstractMatrix{T}}
    mat::M 
    buffer::Vector{T}
end
function ColsOf(x::M) where {T, M<:AbstractMatrix{T}}
    ColsOf{T, M}(x, zeros(T, size(x, 1)))
end
eachcol(x::AbstractMatrix) = ColsOf(x)
Base.start(o::ColsOf) = 1
function Base.next(o::ColsOf, i) 
    for j in eachindex(o.buffer)
        o.buffer[j] = o.mat[j, i]
    end
    o.buffer, i + 1
end
Base.done(o::ColsOf, i) = i > size(o.mat, 2)