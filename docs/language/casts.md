## Casts

A variable of one type can be cast down to a more-specific target type by using a cast function. The name of the cast function is the name of the target type, followed by `?`. The cast function returns an optional of the target type, with a value if the cast was successful, or no value if the cast was unsuccessful.

For `A < B`, with `a:A`, `b:B`, and `c:A?`:

    b <- a;      // OK, as B > A
    c <- A?(b);  // OK, but A < B so must use cast

The optional `c` may be used to check whether the cast succeeded, and if so, to retrieve the result:

    if (c?) {
      f(c!);  // cast was successful, can do something with c
    }

It is possible to cast an optional in the same way, without first checking for a value. The cast of an optional succeeds if that optional has a value, and that value can be cast to the target type. So for `b:B?` instead of `b:B`, the above example is the same.
