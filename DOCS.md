# Getting started

The `birch` driver program can be used to set up a Birch project. From within a (usually empty) directory, run

    birch init --name Example
    
to create a new project, replacing `Example` with the name of the project. This creates the standard files and subdirectories for a Birch project. It is recommended that you maintain this standard structure to make it easier to manage and distribute your project.

Now is a good time to set up version control, such as Git, with this initial set of files.

The standard structure consists of the subdirectories:

  * `bi/` for source code,
  * `build/` for build files,
  * `data/` for input files,
  * `results/` for output files.

and a number of other meta files in the base directory. The most important of these meta files is `META.json`, which contains meta information such as a name, version, and description of the project, and a manifest of source files. As you add files to the project, especially `.bi` source files in the `bi/` subdirectory, you should add them to the `manifest.source` list in `META.json` to include them in the build.

To check for possible issues with the structure, use:

    birch check

To build the project, use:

    birch build
    
To install the project (required to run), use:

    birch install

To run a program that is either part of the standard library or among the source files of the code, use

    birch example
    
replacing `example` with the name of the program. Arguments to the program may be given as command-line options, e.g.:

    birch example -n 10 --input-file data/input.json

When building, Birch will create a number of additional files in the current working directory. Most of these are created in a `build/` subdirectory, although some will appear in the root directory of the project. To delete all of these additional files, use:

    birch clean

Debugging mode is enabled by default, and this will (dramatically) slow down execution times. It is recommended that you keep debugging mode enabled when developing and testing code (perhaps on small problems), but disable it when running tested code (perhaps on serious problems):

    birch build --disable-debug

More information on the `init`, `check`, and `build` programs is available in the documentation of the standard library, and more information on writing your own programs is available in the language guide below.


# The Birch Language

This section assumes that the reader has basic programming experience, but may be arriving at probabilistic programming for the first time. It assumes an understanding of imperative and object-oriented programming paradigms. The introduction is mostly be example.

## Comments

Comments in Birch are as per C++:

    // end-of-line comment
    /* block comment */

In the spirit of JavaDoc and Doxygen, block comments that begin with two stars denote special documentation comments that are extracted by the `birch doc` command:

    /** documentation comment */
    
These typically appear immediately prior to class and function declarations to document their behaviour.

## Variables

A variable `a` of type `A` is declared as follows:

    a:A;

A variable may be given an initial value when declared:

    a:A <- b;

## Types

A thorough treatment of types is deferred to below. There are many categories of types in Birch. Two important categories are:

  1. *Basic types*, such as `Boolean`, `Integer`, `Real`, and `String`.
  2. *Class types* as declared in user code, such as `InputStream`, `OutputStream`, and `Gaussian` from the standard library.

Instantiations of basic types (e.g. variables, expression results) are typically *values*, while those of class types are called *objects*.

## Assignments

Assignment statements use the `<-` operator.

    a <- b;

Assignment is a statement, not an expression; the operator does not return a value:

    a <- b;       // OK!
    a <- b <- c;  // ERROR!

Assignment of basic types is by value, and of class types by reference.

It is possible to declare assignment and conversion operators within a class, allowing assignment to objects from values, or conversion of objects to values, where sensible.

> **Note**
> The operator `=`, often used for assignment in other languages, is reserved for possible future use in Birch (e.g. for declaring equations).

## Control flow

For `a:Boolean` and `b:Boolean`, conditionals can be written as follows:

    if (a) {
      // ...
    } else if (b) {
      // ...
    } else {
      // ...
    }
    
where zero or more `else if` blocks may appear, and zero or one `else` block.

Conditional loops are written as:

    while (a) {
      // ...
    }

or:

    do {
      // ...
    } while (a);

For `a:Integer`, `b:Integer`, and `c:Integer`, a for-loop is written as:

    for (a in b..c) {
      // ...
    }

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

## Sequences

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

> **Note**
> The functionality of sequences is limited at this stage. The primary motivation for their inclusion in the language is for the easy initialization of arrays.

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

## Optionals

Optional types allow variables to have a value of a particular type, or no value at all. An optional type is indicated by following any other type with `?`, like so:

    a:A?;
    
To check whether a variable of optional type has a value, use the postfix `?` operator, which returns `true` if there is a value and `false` if not. If there is a value, use the postfix `!` operator to retrieve it. A common usage idiom is as follows:

    if (a?) {  // check if a has a value
      f(a!);  // if so, do something with the value of a
    }

The special value of `nil` may be assigned to an optional to remove an existing value (if any):

    a <- nil;

> **Note**
> In Birch, a variable of class type always has a value. In some other languages (e.g. Java), variables of class type may have a null value, and this null value is often used to denote no value. In Birch, optionals are always used where a variable may have no value. This is particularly useful when writing functions that accept arguments of class type, as there is no need to check whether those arguments actually have a value or not; they will always have a value, unless denoted as optional.

## Type aliases

The type `A` may be declared as an alias (synonym) for the type `B` with:

    type A = B;

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

## Functions

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

> **Note** Consider the following code:
>
>     fiber iota() -> Integer! {
>       n:Integer <- 0;
>       while (true) {
>         n <- n + 1;
>         yield n;
>     }
>
> This fiber yields the positive integers in order. It never terminates, but need not: the caller will decide how many times it is resumed.

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

## Programs

A program is a special function that is an entry point into Birch code from the command line. It cannot be called from other Birch code. Declare a program with:

    program example(x:Boolean, y:Integer <- 0, message:String,
        long_name:Real) {
      // ...
    }

where `x`, `y`, `message`, and `long_name` are program options, with `y` given a default value of zero. A program has no return value.

To call a program from the command line, use `birch`, followed by the program name, following by a list of program options:

    birch example --message "Hello!" --long-name 10.0 -x true -y 10

Program options may be given in any order but must be named. The name is usually prefixed with double-dash (`--`), but may be prefixed with single-dash (`-`) if the name is a single character. Whenever an underscore (`_`) appears in a name, it should be replaced with a dash (`-`), as in `long_name`/`--long-name` above.

Program options may be of any type for which an assignment from type `String` has been declared. This includes all basic types.

## Operators

Birch supports the most common arithmetic and logical operators found in other programming languages.

### Binary operators

The (infix) binary operators are, in order of highest to lowest precedence:

|     |     |     |     |
| --- | --- | --- | --- |
| `*` Multiply       | `/` Divide      | | |
| `+` Add            | `-` Subtract    | | |
| `<` Less           | `>` Greater     | `<=` Less/equal  | `>=` Greater/equal |
| `==` Equal         | `!=` Not equal  | | |
| `&&` Logical and   | | | |
| `||` Logical or    | | | |

### Unary operators

The (prefix) unary operators are all of equal precedence, and of higher precedence than all binary operators:

|              |              |                 |
| ------------ | ------------ | --------------- |
| `+` Identity | `-` Negative | `!` Logical not |

The standard library provides the obvious overloads for these standard operators for built-in types.

There are no operators for power or modulus: the standard library functions `pow` and `mod` should be used instead. There are no operators defined for bit operations.

### Probabilistic operators

The remaining operators are introduced for concise probabilistic statements. The first is a binary operator that always returns a value of type `Real`, and has precedence less than all of the standard operators:

|              |
| ------------ |
| `~>` Observe |

This operator is syntactic sugar; `a ~> b` is defined to mean exactly:

    b.observe(a);
    
Consequently, it is necessary that `b` is of a class type with an appropriate `observe()` member function defined.

The two remaining probabilistic operators are:

|               |                   |
| ------------- | ----------------- |
| `<~` Simulate | `~` Distribute as |

Like the assignment operator, these operators have no return type, and may only be used in statements, where they have the lowest, and final, precedence.

These are also syntactic sugar; `a <~ b;` means exactly:

    a <- b.simulate();
    
and `a ~ b;` means exactly:

    assert(a.isUninitialized());
    if (!a.isMissing()) {
      yield a ~> b;
    }
    a <- b;
    
If `a` and `b` are expressions, they are evaluated only once, and this code applied to their evaluations.

### Query-Get

These are postfix unary operators used with optional and fiber types. They are of equal precedence, and of higher precedence than all other operators:

|              |              |
| ------------ | ------------ |
| `?` Query    | `!` Get      |

### Overloading

The action of standard operators is defined by overloads, declared using the `operator` statement. Only the standard operators may be overloaded. All other operators have in-built behaviour as described above.

> **Note** It is still possible to manipulate the behaviour of some operators that cannot be overloaded. For example, the behaviour of the assignment operator `<-` can be manipulated by declaring assignments and conversions in class declarations.

A binary operator `+` with two operands `a:A` and `b:B`, and return type `C`, is declared as:

    operator (a:A + b:B) -> C {
      c:C;
      // ...
      return c;
    }
    
Any of the standard binary operators may be used in place of `+`.

A unary operator `+` with one operand `a:A`, and return type `C`, is declared as:

    operator (+a:A) -> C {
      c:C;
      // ...
      return c;
    }
    
Any of the standard unary operators may be used in place of `+`.

Operators always have a return type. It is not possible to manipulate operator precedence.

## Classes

A class named `A` is declared as:

    class A {
      // ...
    }

A variable of class type `A` is then declared as usual:

    a:A;

### Inheritance

A class type named `A` that inherits from a class type named `B` is declared as:

    class A < B {
      // ...
    }
    
The class `B` is referred to as the *super type* of `A`.

### Member variables

Variable declarations that appear within the body of a class are *member variables*:

    class A {
      c:C;
      d:D;
    }

An object of the class type contains instantiations of these member variables, which may be accessed with the dot (`.`) operator:

    f(a.c);
    f(a.d);

### Member functions

Function declarations that appear within the body of a class are *member functions*:

    class A {
      function f(b:B, c:C) -> D {
        // ...
      }
    }

These member functions can be called on an object of the class type, again accessed with the dot (`.`) operator. For `a:A`, `b:B`, `c:C`, `d:D`:

    d <- a.f(b, c);

The body of a member function may use any member variables of the object on which the member function is called. The keyword `this` is used to explicitly refer to the object on which the member function is called. If the class has a super type, the keyword `super` is also used to explicitly refer to the object on which the member function is called, but cast to the super type.

All member functions are virtual. To delegate a call to a member function of the super type, also use the `super` keyword:

    class A < B {
      function f(c:C) {
        super.f(c);  // calls f(c:C) in class B
      }
    }

### Member fibers

Fiber declarations that appear within the body of a class are *member fibers*. Their behaviour is analogous to member functions.

### Generic parameters

A class declaration may include parameters for generic types that are to be specified when the class is used. These are declared using angle brackets in the class declaration:

    class A<T,U> {
      // ...
    }

When a variable of this type is declared, arguments are specified for the generic types, also using angle brackets:

    a:A<B,C>;

These arguments may be of any type. A type argument may be restricted to be some specific type or any subtype of it by using a `<=` operator in the declaration:

    class A<T <= V, U <= Number> {
      // ..
    }

Within the body of the class, the type parameters may be used as though a type themselves:

    class A<T,U> {
      t:T;
      u:U;
      
      function get() -> U {
        return u;
      }
    }

### Initialization parameters

When an object of a class type is declared, its member variables are initialized according to the initial values given in the class body. For the class:

    class A {
      b:Integer <- 0;
      c:Integer;
    }
    
and variable declaration:

    a:A;

The member variables of `a` are initialized such that `a.b == 0`, while `a.c` is uninitialized.

A class can be given initialization parameters, which may be used to initialize any member variables. These are given in parentheses (after any generic parameters, if used):

    class A(d:Integer) {
      b:Integer <- 0;
      c:Integer <- d;
    }

Arguments to these parameters must be given when an object of the class type is declared:

    a:A(1);
    
The member variables of `a` are now initialized such that `a.b == 0`, and `a.c == 1`.

The declarations `a:A;` and `a:A();` are equivalent.

Initialization arguments can be passed onto the super type if required:

    class A(d:Integer) < B(d) {
      // ...
    }

> **Note** Initialization parameters in Birch play a similar role to initialization lists in C++.

Initialization parameters are used for simple object construction, such as to set initial values and array sizes. They do not allow arbitrary code to be executed upon object construction. This is the role of a *constructor*. Birch does not, however, have any special language support for constructors. Instead, it is idiomatic to use *factory functions*, exploiting the fact that the same name can be used for both a function and a class in the Birch language.

A factory function is given the same name as the class it is intended to construct:

    function A(b:B, c:C) -> A {
      a:A;
      // ...
      return a;
    }

This function is treated as any other---there is nothing special about it---but it is idiomatic that such a function should return an object of the same type as its name, or of a subtype of that type. The possibility of returning a subtype makes a factory function slightly more flexible than an ordinary constructor.

For complex object construction, it can be useful to define a member function within the class that does most of the work, with the factory function simply instantiating the object, then passing its arguments to this function. It is idiomatic for such a member function to be given the name `make`.

### Assignments

> **Note** Recall that, for basic types, assignment is by value, while for class types, assignment is by reference.

Objects of class type `A` may be assigned another object of type `A` or an object of any subtype of `A`; i.e. if `a:A` and `b:B` with `A < B`, it is possible to assign `b <- a` but not `a <- b`.

Such assignments are by reference. Objects of class type `A` may be assigned *by value* if an appropriate declaration has been made within the class body. To permit assignment of type `C`, for example:

    class A {
      operator <- c:C {
        // ...
      }
    }
    
The body of the operator should update the state of the object using the argument. There is no return value. For `a:A` and `c:C`, the assignment `a <- c` would then be valid, even though `C` is not a subtype of `A`.

### Conversions

Objects of class type `A` may be implicitly cast to an object of any super type of `A`; i.e. if `a:A` and `b:B` with `A < B`, the object `a` can be implicitly converted to an object of type `B`, as in the following:

    function f(b:B) {
      // ...
    }
    a:A;
    f(a);
    
Such casts are *by reference*. For other types, it is possible to declare implicit conversions *by value*, if an appropriate declaration has been made within the class body. To permit conversion to type `C`, for example:

    class A {
      operator -> C {
        c:C;
        // ...
        return c;
      }
      ...
    }
    
The body of the operator should construct the object to be returned as the result of the conversion. For `a:A` and `c:C`, the following would then be valid, even though `C` is not a super type of `A`

    function f(c:C) {
      // ...
    }
    
    a:A;
    f(a);

## Special Topics

### Nested C++ code

The Birch language permits raw C++ code to be nested. Nesting C++ code requires some understanding of how the Birch compiler translates Birch code to C++ code, and this translation is still in flux.

The primary motivation for nested C++ is to wrap Birch language interfaces around existing C++ language libraries. The Birch standard library, for example, uses this feature to integrate parts of the C++ Standard Template Library and Eigen linear algebra library. The use of nested C++ code for other purposes is discouraged, and it is recommended that its use is kept to a minimum, with any lengthy or complex nested C++ code separated into dedicated header and source files.

When building a package, the Birch compiler creates one `.hpp` file per package, and one `.cpp` file per source.

To include raw C++ code in the `.hpp` file, use the following, typically at the root scope of a `.bi` file:

    hpp{{
    // C++ code here
    }}

This is useful for `#include` directives, or for declaring C++ types and functions.

To include raw C++ code in the `.cpp` file, use the following, typically in the body of a function in a `.bi` file:

    cpp{{
    // C++ code here
    }}

This is useful for executing arbitrary C++ code. It is beyond the scope of this document to detail all the ways in which such code may interface with C++ code generated by the Birch compiler. The following rules cover common use cases, however:

  * Variables declared in Birch code may be used by appending `_` to their declared name. Likewise functions and types, where it may also be necessary to prepend the name with a `bi::` or `bi::type::` namespace designator.
  * Variables of class type declared in Birch code are accessed via custom smart pointers C++ code.
  * Nesting C++ code within fibers requires special care. Variables declared in  nested C++ code will not be preserved between yields. One option is to wrap these as member variables within a Birch object, as the object is preserved between yields.

For examples of nested C++ code, see the Birch standard library. For an in-depth study, see the C++ code generated by the Birch compiler in the `build/` subdirectory of your Birch project, the `libbirch/` subdirectory within the Birch compiler sources, and the C++ code generator within the `bi/io/` subdirectory of the Birch compiler sources.

# The Birch Driver

This section documents the `birch` driver program, used for building and running Birch code. The driver program is partly controlled by command-line arguments, and partly by the `META.json` file included in the directory in which it is run. These are documented in turn.

## Commands

The driver program is invoked as follows:

    birch <command> [options]
    
Each command is documented below, along with the options it accepts.

### init

    birch init [options]

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `Untitled`).

### check

    birch check

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in `META.json` that do not exist,
  - files of recognisable types that exist but are not listed in
    `META.json`, and
  - standard meta files that do not exist.

### build

    birch build [options]

Build the project.

  - `--include-dir` : Add search directory for header files.
  - `--lib-dir` : Add search directory for library files.
  - `--share-dir` : Add search directory for data files.

These three options are analogous to their counterparts for a C/C++ compiler, and specify the locations in which the Birch compiler should
search for headers (both Birch and C/C++ headers), installed libraries and
installed data files. They may be given multiple times to specify multiple
directories in the order in which they are to be searched.

After searching these directories, the Birch compiler will search the environment variables `BIRCH_INCLUDE_PATH`, `BIRCH_LIBRARY_PATH` and `BIRCH_SHARE_PATH`, followed by the directories of the compiler's own installation, followed by the system-wide locations `/usr/local/` and
`/usr/`.

  - `--prefix` : Installation prefix (default platform-specific).
  - `--enable-std` / `--disable-std` : Enable/disable the standard library.
  - `--enable-warnings` / `--disable-warnings` : Enable/disable warnings.
  - `--enable-debug` / `--disable-debug` : Enable/disable debug mode.
  - `--enable-verbose` / `--disable-verbose` : Verbose mode.

### install

    birch install

Install the project. This installs all header, library and data files needed by the project into the directory specified by `--prefix` (or the system default if this was not specified).

### uninstall

    birch uninstall

Uninstall the project. This uninstalls all header, library and data files from the directory specified by `--prefix` (or the system default if this was not specified).

### docs

    birch docs

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory.

> **Note** The Birch documentation system is inspired by JavaDoc and Doxygen. It is suggested to use it similarly.

It will be overwritten if it already exists, and may be readily converted to other formats using a utility such as `pandoc`.

The content of `DOCS.md` is gathered from documentation comments that precede declarations:

    /**
     * Documentation comment.
     */
     class A {
       // ...
     }

    /**
     * Documentation comment.
     */
    function f(a:A, b:B) {
      // ...
    }
     
    /**
     * Documentation comment.
     */
    a:A;
     
While the content of these documentation comments is not prescribed, the format should be Markdown, as they are copied verbatim into the `DOCS.md` file where required. It is suggested that the first sentence of each comment is a brief, standalone description, and that parameters are documented using a bulleted list as follows:

    /**
     * Do something.
     * 
     * - a: The first parameter.
     * - b: The second parameter.
     */
    function f(a:A, b:B) {
      // ...
    }

### dist

    birch dist

Build a distributable archive for the project. This creates an archive file of the name `Example-x.y.z.tar.gz` in the working directory, where `Example` is the name of the project and `x.y.z` the version number.

### clean

Clean the project directory of all build files.

## Meta file

Each Birch project contains a `META.json` file providing a name, version and description of the project, as well as a manifest of its files and dependencies. The file may contain the following keys.

name
: The name of the project (as a string).

version
: The version of the project (as a string).

description
: A one-sentence description of the project (as a string).

manifest
: An object containing the following keys, listing the files to be included in the project.

manifest.source
: A list of source files (as strings) to be compiled. These typically have a `.bi` file extension. If C/C++ sources are included, they may have `.c` or `.cpp` file extensions.

manifest.header
: A list of header files (as strings) to be installed. This is typically only used if C/C++ sources are included, in which case these files are likely to have `.h` or `.hpp` file extensions.

manifest.data
: A list of data files (as strings) to be installed. These are typically input files provided by the project.

manifest.other
: A list of other files (as strings) to be distributed but not installed. These are typically meta files, such as `README.md`, `LICENSE`, and `META.json` itself.

require
: An object containing the following keys, describing the dependencies of the project. Their presence is checked during the build of the project.

require.header
: A list of header files (as strings). This is typically only used for C/C++ dependencies, in which case these files are likely to have `.h` or `.hpp` file extensions.

require.library
: A list of names of libraries (as strings). This is typically only used for C/C++ library dependencies.

require.program
: A list of names of programs (as strings).
