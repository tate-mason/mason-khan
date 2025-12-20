using LinearAlgebra, Statistics

"""
Basics of arrays
"""

a = [10, 20, 30] # 1D array of Int64
a = [1.0, 2.0, 3.0] # 1D array of Float64
typeof(randn(100)) # 1D array of Float64

ndims(a) # number of dimensions
size(a)  # size of each dimension

# Array vs Matrix
Array{Int64, 1} == Vector{Int64} # true
Array{Int64, 2} == Matrix{Int64} # true

[1, 2, 3] == [1; 2; 3]
[1 2 3] # 2d row vector

"""
Julila has both 1d arrays of size (1, n) or (n, 1) (row vs column vectors)

Need both for linear algebra operations
"""

# Creating Arrays

## Array creating functions

zeros(3) # 1D array of zeros
zeros(2, 2)

fill(5.0, 2, 2) # return an array filled with 5.0

x = Array{Float64}(undef, 3) # uninitialized 1D array of Float64
fill(0, 2, 2) # 2D array of Int64 filled with 0

fill(false, 2, 2) # 2D array of Bool filled with false

## Creating Arrays from Existing Arrays

x = [1, 2, 3]
y = x
y[1] = 2
x

# y = x creates a new binding called y, returning whatever x binds to.

x = [1, 2, 3]
y = copy(x)
y[1] = 2
x

# copy(x) creates a new array in memory, copying the contents of x into it.

x = [1, 2, 3]
y = similar(x)
y
# y is a similarly sized array to x

x = [1, 2, 3]
y = similar(x, 4)
# y is a new array of length 4, with the same element type as x

x = [1, 2, 3]
y = similar(x, 2, 2)
# y is a new 2x2 array with the same element type as x

## Array Definitions from Literals

a = [10, 20, 30, 40] # 1D array of Int64
a = [10 20 30 40] # 1 x n 2D array of Int64
ndims(a)
a = [10 20;
     30 40] # 2 x 2 2D array of Int64
a = [10; 20; 30; 40] # n x 1 2D array of Int64
ndims(a)

a = [10 20 30 40]' # transpose of a 1 x n array is an n x 1 array
ndims(a)

"""
Create multidimensional arrays from literals with ;
"""

[1; 2] # create vector

[1; 2 ;;] # add extra dimension with ;
[1; 2 ;;;] # add yet another extra dimension with ;;;

[1;;] # 1x1 matrix

[1 2; 3 4;;;] # 2x2x1 array

"""Array Indexing
"""

a = [10 20 30 40]
a[end - 1]

a[1:3] # slicing
a = randn(2, 2)
a[1, 1] # indexing into 2D array

a[1, :] # first row
a[:, 1] # first column

a = randn(2, 2)
b = [true false; false true]
a[b] # boolean indexing

a = zeros(4)
a[2:end] .= 42 # broadcasting assignment

a # modified array

"""
Views and Slices
"""

a = [1 2; 3 4]
b = a[:, 2]
@show b # b is a copy of the second column of a
a[:, 2] = [4, 5] # modify a only
@show a
@show b; # b remains unchanged

## Views
a = [1 2; 3 4]
b = view(a, :, 2) # create a view into the second column of a, no copy made
@show b
a[:, 2] = [4, 5] # modify a
@show a
@show b; # b reflects the change in a

@views b = a[:, 2] # create a view into the second column of a
view(a, :, 2) == b # true

## @views are not normal dense arrays

a = [1 2; 3 4]
b_slice = a[:, 2]
@show typeof(b_slice)
@show typeof(a)
@views b = a[:, 2]
@show typeof(b);

a = [1 2; 3 4]
b = a'
typeof(b)

a = [1 2; 3 4]
b = a'
c = Matrix(b) # convert to dense matrix
d = collect(b) # `collect` works on any iterable
c == d # true

"""
Special Matrices
"""

d = [1.0, 2.0]
a = Diagonal(d) # create a diagonal matrix from vector d

@show 2a
b = rand(2, 2)
@show b * a;
