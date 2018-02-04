## Lambdas

Lambda (anonymous) functions may appear in expressions as:

    @(a:A, b:B) -> C {
      c:C;
      // ...
      return c;
    }

The type of such a lambda is `@(A, B) -> C`. It is possible to declare a variable of this *function type*:

    f:@(A, B) -> C;

to assign values to it:

    f <- @(a:A, b:B) -> C {
          c:C;
          // ...
          return c;
      };

and to call it:

    f(a, b);

Functions can accept lambdas as arguments. Such a function may be declared:

    function g(f:@(a:A, b:B) -> C) -> D {
      d:D;
      // ...
      return d;
    }

and be called with:

    g(@(a:A, b:B) -> C {
          c:C;
          // ...
          return c;
        });

or, if the lambda was previously assigned to a variable `f:@(a:A, b:B) -> C` as above:

      g(f);
