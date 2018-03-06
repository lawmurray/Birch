A function with two parameters `a:A` and `b:B`, and return type `C`, is declared as:

    function f(a:A, b:B) -> C {
      c:C;
      // ...
      return c;
    }

For a function without a return value, omit the `->`:

    function f(a:A, b:B) {
      // ...
    }

For a function without parameters, use empty parentheses:

    function f() {
      // ...
    }

Functions are called by giving arguments in parentheses:

    f(a, b)

When calling a function without parameters, use empty parentheses:

    f()
