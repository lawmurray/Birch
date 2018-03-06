Sequences are constructed using square brackets:

    [a, b, c]

For `a:A`, `b:A`, and `c:A`, the type of such a sequence is `[A]`.

Generally, for `a:A`, `b:B`, and `c:C`, the type of such a sequence is `[D]`, where `D` is the least common super type of `A`, `B`, and `C`. Such a super type must exist for a sequence to be valid.

It is possible to declare a variable of the *sequence type*:

    d:[D];

and to assign values to it:

    d <- [a, b, c];

It is possible to nest sequences:

    [[a, b, c], [e, f, g]]

If the type of the inner sequences is `[D]`, then the type of the outer sequence is `[[D]]`; i.e. it is a sequence of sequences of `D`.

It is not possible to access the individual elements of a sequence, either for reading or writing. To access the individual elements, assign the sequence to an array, and access them via the array.

!!! note
    The functionality of sequences is limited at this stage. The primary motivation for their inclusion in the language is for the easy initialization of arrays.
