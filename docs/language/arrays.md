## Arrays

An array `a` with elements of type `A` is declared as:

    a:A[_];

Multidimensional arrays, with any number of dimensions, are declared as:

    b:B[_,_];    // two-dimensional array
    c:C[_,_,_];  // three-dimensional array, etc

The rightmost dimension is the innermost (fastest moving) in the memory layout. Two-dimensional arrays that represent matrices are therefore in row-major order.

The size of the array may be given in the square brackets when it is declared, in place of `_`:

    a:A[4];
    b:B[4,8];

Arrays are sliced with square brackets. To select the element of `b` at row 2 and column 6, use:

    b[2,6]

This returns a single element of type `B`. To select the range of elements of `b` at row 2 and columns 5 to 8, use:

    b[2,5..8]

This returns a vector of type `B[_]`.

In the context of array slicing, the term *index* denotes a single index, as in `2` and `6` above; while the term *range* denotes a pair of indices separated by `..`, as in `5..8` above.

Indices reduce the number of dimensions in the result; they do not create singleton dimensions. Revisiting the previous example for emphasis, the result is of type `B[_]` with size 4, not of type `B[_,_]` with size 1 by 4. When a singleton dimension is desired, use a singleton range that starts and ends at the same index:

    b[2..2,5..8]

Arrays are resized by assignment, e.g.

    a:A[4];
    d:A[2];
    d <- a;

The vector `d` is now a copy of `a`, with size 4. Its previous value is discarded.

When slicing an array on the left side of an assignment, suggesting a view of the existing array, sizes must match on the left and right:

    d[1..2] <- a[1..2];  // OK! Both left and right have size 2
    d[1..2] <- a;          // ERROR! Left has size 2, right has size 4

Assignment may be used to resize an array, but not to change its number of dimensions. The number of dimensions of an array is a fundamental part of its type.

Sequences can be assigned to arrays:

    x <- [a, b, c];
    x <- [[a, b, c], [d, e, f]];

But arrays cannot be assigned to sequences, as sequences are read-only.
