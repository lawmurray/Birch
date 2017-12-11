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

## Types

A thorough treatment of types is deferred to below. There are many categories of types in Birch. Two important categories are:

  1. *Basic types*, such as `Boolean`, `Integer`, `Real`, and `String`.
  2. *Class types* as declared in user code, such as `InputStream`, `OutputStream`, and `Gaussian` from the standard library.

## Variables

A variable `x` of type `X` is declared as follows:

    x:X;

## Assignments

Assignment statements use the `<-` operator.

    x <- y;
    
A variable may be given an initial value when declared:

    x:X <- y;

Assignment is a statement, not an expression; the operator does not return a value:

    x <- y;       // OK!
    x <- y <- z;  // ERROR!

Assignent of objects of basic type is by value, while those of class type is by reference.

It is possible to declare assignment and conversion operators within a class, allowing assignment of objects of basic type, or conversion to an object of basic type, where sensible.

> The operator `=`, sometimes used for assignment in other languages, is currently reserved for future use (e.g. for declaring equations).

## Tuples

Tuples are tied using parentheses:

    (x, y, z)
    
For `x:X`, `y:Y`, and `z:Z`, the type of such a tuple is `(X, Y, Z)`.

It is possible to declare a variable of the *tuple type*:

    a:(X, Y, Z);
    
and to assign values to it:

    a <- (x, y, z);
    
To untie a tuple, use parentheses on the left:

    (x, y, z) <- a;

## Sequences

Sequences are constructed using square brackets:

    [x, y, z]
    
For `x:X`, `y:Y`, and `z:Z`, the type of such a sequence is `[W]`, where `W` is the least common super type of `X`, `Y`, and `Z`. Such a super type must exist. In the simplest case, `x`, `y` and `z` are all of the same type `W`, and so the sequence is of type `[W]`.

It is possible to declare a variable of the *sequence type*:

    w:[W];

and to assign values to it:

    w <- [x, y, z];
    
It is possible to nest sequences:

    [[x, y, z], [a, b, c]]
    
If the least common subtype of all elements is `W`, this is a sequence of sequences of `W`, with the type `[[W]]`.

It is not possible to access the individual elements of a sequence, either for reading or writing. To access the individual elements, assign the sequence to an array, and access them via the array.

> In future, it may be possible to use a `for` loop to iterate over the elements of a sequence for read-only access. The primary  motivation for sequences in the language is for the easy initialization of arrays, however.

## Arrays

An array `x` with elements of type `X` is declared as:

    x:X[_];
    
Multidimensional arrays, with any number of dimensions, are declared as:

    x:X[_,_];    // two-dimensional array
    x:X[_,_,_];  // three-dimensional array, etc

The rightmost dimension is the innermost (fastest moving) in the memory layout. Two-dimensional arrays that represent matrices are therefore in row-major order.

The size of the array may be given in the square brackets when it is declared, in place of `_`:

    x:W[4];
    y:W[4,8];

Array slicing is with square brackets. To select the element of `y` at row 2 and column 6, use:

    y[2,6]

This returns a single element of type `W`. To select a range of elements of `y` at row 2 and columns 5 to 8, use:

    y[2,5..8]
    
This returns a vector of type `W[_]`.

In the context of array slicing, the term *index* denotes a single index, as in `2` and `6` above; the term *range* denotes a pair of indices separated by the `..` operator, as in `5..8` above.

Indices reduce the number of dimensions in the result, as opposed to creating singleton dimensions. Thus, in the last example, the result is of type `W[_]` with size 4, not of type `W[_,_]` with size 1 by 4. If a singleton dimension is desired, a range can be used:

    y[2..2,5..8] 

Arrays are resized by assignment, e.g.

    z:W[2];
    z <- x;
    
The vector `z` is now a copy of `x` with a size of 4. Assignment may be used to resize an array, but not to change its number of dimensions.

When using a slice on the left size of an assignment, array sizes must match on the left and right:

    z[1..2] <- x[1..2];  // OK! Both left and right have size 2
    z[1..2] <- x;          // ERROR! Left has size 2, right has size 4

Sequences can be assigned to arrays, but not vice versa:

    x <- [a, b, c];
    x <- [[a, b, c], [d, e, f]];

## Optionals

A variable of class type must always point to an object. When declared, a variable of class type must be constructed:

    a:A(b, c);
    
or be assigned a value:

    a:A <- b;

If the constructor for the class type takes no parameters, the parentheses may be omitted:

    a:A;

but this is equivalent to:

    a:A();

and an object is constructed in this case.

Some use cases require that a variable may not have a value. *Optional types* are provided for this purpose. An optional type is indicated by following any other type with the `?` type operator, like so:

    a:A?;
    
To check whether a variable of optional type has a value, use the postfix `?` operator, which returns a `Boolean` giving `true` if there is a value. To get that value, use the postfix `!` operator, which returns a value of the original type.

A common usage idiom is as follows:

    a:A?;
    ...
    if (a?) {  // a? gives a Boolean
      f(a!);  // when a? is true, a! gives an A, otherwise an error
    }

## Casts

## Conditionals

    if (condition) {
      ...
    } else if (condition) {
      ...
    } else {
      ...
    }

## Loops

    while (condition) {
      ...
    }

    for (n in from..to) {
      ...
    }

## Functions

Functions are called by giving arguments in parentheses:

    f(a, b)

A function with two parameters (of type `A` and `B`) and return type `C` is declared as follows:

    function f(a:A, b:B) -> C {
      c:C;
      ...
      return c;
    }

For a function without a return, omit the right-arrow syntax:

    function f(a:A, b:B) {
      ...
    }

For a function without parameters, empty parentheses are required:

    function f() {
      ...
    }

## Programs

A program represents an entry point into Birch code from the command line or other host software. It cannot be called from other Birch code. It is declared as follows:

    program example(x:Boolean, y:Integer <- 0, message:String,
        long_name:Real) {
      // ...
    }

where `x`, `y` and `message` are program options, with `y` given a default value of zero.

The `birch` driver can then be used from the command line to call this program:

    birch example -x true -y 10 --message "Hello world!" --long-name 10.0

Program options may be given in any order but must be named. The name is usually prefixed with a double dash (`--`), but can be prefixed with a single dash (`-`) in the case that it is a single character.

Program option names that contain an underscore are specified with a dash on the command line, as in `long_name`/`--long-name` above.

Program options may be of any type for which an assignment from type `String` has been declared. This includes all basic types.

## Fibers

A fiber is like a function for which execution can be paused and later resumed. Unlike a function, which can return just one value, a fiber yields a value each time it is paused.

A fiber with two parameters (of type `A` and `B`) and yield type `C` is declared as follows:

    fiber f(a:A, b:B) -> C! {
      c:C;
      ...
      yield c;
    }

Note the decoration of the yield type `C` with `!` to make it a fiber type `C!`. This is required.

When called, a fiber performs no execution except to construct an object of fiber type `C!` and return it; execution of the body of the fiber is then controlled through that object.

The usage idiom for fibers is similar to optionals, but with a loop:

    c:C! <- f(a, b);
    while (c?) {
      g(c!);  // do something with the yield value
    }
    
It is the postfix `?` operator that triggers the continuation of the fiber execution, which proceeds until the next yield point. Repeated use of the postfix `!` operator between calls of the postfix `?` operator will retrieve the last yield value, without further execution.

Within the body of a fiber, the `yield` statement is used to pause execution. This yields execution to the caller, along with the given value.

The fiber terminates when execution reaches the end of the body, if ever. When terminating, it does not yield a value. To terminate the execution of a fiber before reaching the end of the body, use an empty `return;` statement.

### Fibers-within-fibers

An outer fiber may call an inner fiber using the above syntax, as if the outer fiber were any other function. For convenience, the following implicit behaviour is also specified.

If the outer fiber calls the inner fiber in such a way that its return value is ignored:

    f(a, b);

this implicitly behaves as:

    c:C! <- f(a, b);
    while (c?) {
      yield c!;
    }

That is, the outer fiber yields the values of the inner fiber until the inner fiber completes execution.

The same behaviour applies to an outer fiber that calls an inner *function*---not itself a fiber---which has a fiber return value.

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
    
and, in fact, is internally transformed to this on use. Consequently, it is necessary that `b` is of a class type with an appropriate `observe()` member function defined.

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

A higher-level treatment of these operators is given in later sections, which is more meaningful for a reader who is approaching Birch for the first time.

### Query-Get

These are postfix unary operators used with optional and fiber types. They are of equal precedence, and of higher precedence than all other operators:

|              |              |
| ------------ | ------------ |
| `?` Query    | `!` Get      |

See below for the behaviour of these operators.

### Overloading

The action of standard operators is defined by overloads in Birch code, declared using the `operator` statement.

The precedence of operators is always the same. It cannot be manipulated by Birch code.

Only the standard operators may be overloaded. All other operators have in-built behaviour as described above. It is still possible to manipulate the behaviour of some operators that cannot be overloaded. For example, the behaviour of the assignment operator `<-` can be manipulated by declaring assignments and conversions in class declarations, as described below.

A binary operator `+` with two operands (of type `A` and `B`) and return type `C` is overloaded as follows:

    operator a:A + b:B -> C {
      c:C;
      ...
      return c;
    }
    
Any of the standard binary operators may be used in place of `+`.

A unary operator `+` with one operand (of type `A`) and return type `C` is declared as follows:

    operator +a:A -> C {
      c:C;
      ...
      return c;
    }
    
Any of the standard unary operators may be used in place of `+`.

Operators always have a return type. It is not possible to manipulate operator precedence.

## Classes

A class type named `Base` is declared as follows:

    class Base {
      ...
    }

A class named `Derived` that inherits from `Base` is declared as follows:

    class Derived < Base {
      ...
    }
    
### Member variables

### Member functions

From within a member function, it is possible to use the keyword `this` to refer to the current object, e.g.

    function f(a:A) {
      this.a <- a;
    }

All member functions are virtual. It is possible to delegate to a member function in the most-immediate base type by using the `super` keyword, e.g.

    function f(a:A) {
      super.f(a);
      ...
    }

### Member fibers
    
All member fibers are virtual.
    
### Constructors

If the constructor of `Base` is parameterized, the parameters of the constructor are declared as follows:

    class Base(a:A, b:B) {
       ...
    }
    
A variable of type `Base` should provide arguments for these parameters when declared:

    o:Base(a, b);

Constructor parameters may be passed to a base type and used to initialize other member variables:

    class Derived(a:A:, b:B) < Base(a) {
      c:C(a, b);
      d:D(a, b);
      ...
    }

Parameters of the constructor are also private member variables, and may be used by member functions.

### Assignments

Objects of class type `Base` may be assigned an object of type `Base` or of any type that derives from `Base`. For example, if `Derived` derives from `Base`:

    o1:Base;
    o2:Derived;
    
it is possible to assign `o1 <- o2` but not `o2 <- o1`.

Such assignments are *by reference*. For example, after executing `o1 <- o2`, both `o1` and `o2` point to the same object.

For other types, it is possible to declare assignments *by value*. This is done by including a member operator in the class declaration. To permit value assignment of objects of type `A`, for example:

    class Base {
      operator <- a:A {
        ...
      }
      ...
    }
    
The body of the operator should update the state of the object for a value assignment from the argument. The following assignment would then be valid:

    o:Base;
    a:A;
    o <- a;

### Conversions

Objects of class type `Derived` may be treated as an object of type `Derived`, or of any base type of `Derived`. For example, if `Derived` derives from `Base`, an object of type `Derived` may be passed to a function that expects an argument of type `Base`:

    function f(o:Base) {
       ...
    }
    
    o:Derived;
    f(o);
    
Such conversions are *by reference*.

For other types, it is possible to declare conversions *by value*. This is done by including a member operator in the class declaration. To permit value conversion of objects of type `A`, for example:

    class Derived {
      operator -> A {
        a:A;
        ...
        return a;
      }
      ...
    }
    
The body of the operator should construct the object that is the result of the conversion. The following conversion would then be valid:

    function f(a:A) {
      ...
    }
    
    o:Derived;
    f(o);

## Generics

# Special Topics

## Build system

## Documentation system

## Nested C++ code
