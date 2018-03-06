Assignment statements use the `<-` operator.

    a <- b;

Assignment is a statement, not an expression; the operator does not return a value:

    a <- b;       // OK!
    a <- b <- c;  // ERROR!

Assignment of basic types is by value, and of class types by reference.

It is possible to declare assignment and conversion operators within a class, allowing assignment to objects from values, or conversion of objects to values, where sensible.

!!! note
    The operator `=`, often used for assignment in other languages, is reserved for possible future use in Birch (e.g. for declaring equations).
