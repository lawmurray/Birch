## Tuples

Tuples are tied using parentheses:

    (a, b, c)

For `a:A`, `b:B`, and `c:C`, the type of such a tuple is `(A, B, C)`.

It is possible to declare a variable of the *tuple type*:

    d:(A, B, C);

and to assign values to it:

    d <- (a, b, c);

To untie a tuple, use parentheses on the left:

    (a, b, c) <- d;
