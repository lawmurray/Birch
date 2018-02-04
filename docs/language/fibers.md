## Fibers

A fiber works similarly to a function, but its execution can be paused and resumed. Where a function, when called, executes to completion and may *return* a value to the caller, a fiber, when resumed, executes to its next pause point, and may *yield* a value to the caller. The caller can then proceed with some other computation, and later resume the fiber again, at which point it will execute to its next yield point. The state of a fiber is maintained between the pause and resume, so that it always resumes execution from where it last paused.

A fiber is declared with:

    fiber f(a:A, b:B) -> C! {
      c:C;
      // ...
      yield c;
    }

where `C` is the yield type and `C!` the *fiber type*. The `yield` statement is used to pause execution and yield a value to the caller, analogous to the `return` statement for functions. The fiber terminates when execution reaches the end of the body. When terminating, it does not yield a value. To terminate the execution of a fiber before reaching the end of the body, use an empty `return;` statement.

When called, a fiber performs no execution except to construct a value of the fiber type and return it. The execution of the fiber is controlled via this value. The usage idiom for controlling fibers is analogous to optionals, except that where an optional has zero or one value, a fiber has zero or more values. The if-statement for optionals is replaced with a while-loop for fibers:

    c:C! <- f(a, b);
    while (c?) {
      g(c!);  // do something with the yield value
    }

It is the query operator (`?`) that resumes the fiber. The fiber then continues execution until it yields another value, or it terminates. If it yields a value, the query operator returns true to the caller, otherwise it returns false to the caller. If returns true, the get operator (`!`) is then used to retrieve the yield value. Repeated use of the get operator between calls to the query operator will retrieve the last yield value without further execution.

It is not necessary for the caller to run the fiber to termination. Likewise, it is not necessary for a fiber to ever terminate, which may be a design choice.

!!! note
    Consider the following code:

        fiber iota() -> Integer! {
          n:Integer <- 0;
          while (true) {
          n <- n + 1;
          yield n;
        }

    This fiber yields the positive integers in order. It never terminates, but need not: the caller will decide how many times it is resumed.

Fibers may call other fibers. No special syntax is required for this. There is a common use case, however, where a fiber with yield type `C` calls another fiber with yield type `C`, and wishes the pass the yield values of that second fiber back to the original caller. For the convenience of this common use case, the following implicit behaviour is defined.

If a fiber calls another fiber, but ignores the return value that would otherwise be used to control the execution of that fiber:

    f(a, b);

this implicitly behaves as though the following were written:

    c:C! <- f(a, b);
    while (c?) {
      yield c!;
    }

That is, the second fiber yields values to the first fiber, which in turn yields those values back to its caller.

The same implicit behaviour applies when a fiber calls a *function*---not itself a fiber---but that returns a fiber type.
