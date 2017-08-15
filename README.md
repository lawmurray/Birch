# Introduction

Birch is a compiled, imperative, object-oriented, and probabilistic programming language. The latter is its primary research concern. The Birch compiler uses C++ as a target language.

# Installation

## Installing from Git

If you have acquired Birch directly from its Git repository, first run the following command from within the `Birch` directory:

    ./autogen.sh
    
then follow the instructions for *Installing from source*.

## Installing from source

Birch requires the Boost libraries, Eigen linear algebra library and Boehm garbage collector (`libgc`). These should be installed first.

To build and install, run the following from within the `Birch` directory:

    ./configure
    make
    make install
    
This installs three components:

  1. `birch` (the driver program),
  2. `birchc` (the compiler program), and
  3. `libbirch.*` and associated `bi/*.hpp` header files (the compiler library).

Typically, only the first of these is used directly. It provides a friendly wrapper for building and running Birch code, calling the compiler program, and linking in the compiler library, where appropriate. It is usually unnecessary to become familiar with the latter two.

### Installing the standard library

You will also want to install the standard library. This provides all the standard functionality used by Birch programs, such as mathematics functions, probability distributions, and I/O. It is in a separate `Birch.Standard` repository. To build and install, run the following from within the `Birch.Standard` directory:

    make
    make install
    
Note that these, in turn, are just calling the `birch` driver program to build and install the code.

### Installing the examples

You may also want to install the example programs. These are in a separate `Birch.Example` repository. To build and install, run the following from within the `Birch.Example` directory:

    make
    make install

Note that these, in turn, are just calling the `birch` driver program to build and install the code.
    
To run an example, use:

    birch example

replacing `example` with the name of the example program (see the sources in the `bi/` directory for programs and their options).

# Getting started

The `birch` driver program can be used to set up a Birch project. From within a (usually empty) directory, run

    birch init --name Example
    
to create a new project, replacing `Example` with the name of your project. This creates a standard layout for a Birch project. It is recommended that you maintain this standard layout to make it easier to manage and distribute your project.

Now is a good time to set up your version control, such as Git, with this initial set of files.

The standard directory structure consists of the subdirectories:

  * `bi/` for source code,
  * `data/` for data files,
  * `build/` for build files,

and a number of other meta files in the base directory. The most important of these meta files is `MANIFEST`, which contains a list of all source files, as well as other files that should be distributed with the project (e.g. data files). As you add files to the project, especially `.bi` source files in the `bi/` subdirectory, you should add them to the `MANIFEST` file to include them in the build.

To check for possible issues with the `MANIFEST` file, use:

    birch check

To build the project, use:

    birch build
    
This will create a number of additional files in the current working directory, required for the build. Most of these are created in a `build/` subdirectory to keep them out of the way as much as possible.

Note that debugging mode is enabled by default, and this will (dramatically) slow down execution times. It is recommended that you keep debugging mode enabled when developing and testing code on small problems, but disable it when running tested code on serious problems:

    birch build --disable-debug

To run a program that is either part of the standard library or amongst the project code, use

    birch example
    
replacing `example` with the name of the program. Arguments to the program may be given as command-line options, e.g.:

    birch example -N 10 -T 10

More information on the `init`, `check`, and `build` programs is available in the documentation of the standard library, and more information on writing your own programs is available in the language guide below.


# The Birch Language

This section assumes that the reader has basic programming experience, but may be arriving at probabilistic programming for the first time. It assumes an understanding of imperative and object-oriented programming paradigms. Syntax is given by example, while the semantics are as an experienced programmer will expect and so not detailed.

Greater focus is given to the probabilistic programming paradigm, which may be less familiar. This includes more detail on programming concepts that the reader may not have encountered, but that play an important role in Birch, such as fibers and multimethods.

## Programs

A program represents an entry point into Birch code from the command line or other host software. It cannot be called from other Birch code. It is declared as follows:

    program example(a:Real, b:Integer = 0, msg:String) {
      ...
    }

where `a`, `b` and `msg` are program options, with `b` given a default value.

The `birch` driver can then be used from the command line to call this program:

    birch example -a 10.0 -b 10 --msg "Hello world!"

Note that program options may be given in any order but must be named. The name is usually prefixed with a double dash (`--`), but can be prefixed with a single dash (`-`) instead if the name is only one character long.

Program options may be of any built-in type, or any class type for which an assignment from type `String` has been declared.

## Variables

A variable `a` of type `A` is declared as follows:

    a:A;

Basic types, provided by the standard library, include `Boolean`, `Integer`, `Real` and `String`.

## Assignments

Assignment statements use the `<-` operator.

    a <- b;

Behaviour depends on assignment and conversion declarations in any class types involved. In the absence of these, assignment of objects of basic type is by value, and of class type by reference.
    
An assignment has no return type, and as it is a statement, not an expression. That is, the line of code `a <- b;` is valid, while `a <- b <- c;` is not, as assignment operators cannot be chained together in expressions.

Note that the assignment operator is always `<-`. The operator `=`, sometimes used for assignment in other languages, is currently reserved for future use (e.g. for declaring equations).

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

| ------------ | ------------ | --------------- |
| `+` Identity | `-` Negative | `!` Logical not |

The standard library provides the obvious overloads for these standard operators for built-in types.

There are no operators for power or modulus: the standard library functions `pow` and `mod` should be used instead. There are no operators defined for bit operations.

### Probabilistic operators

The remaining operators are introduced for concise probabilistic statements. The first is a binary operator that always returns a value of type `Real`, and has precedence less than all of the standard operators:

| ------------ |
| `~>` Observe |

This operator is syntactic sugar; `a ~> b` is defined to mean exactly:

    b.observe(a);
    
and, in fact, is internally transformed to this on use. Consequently, it is necessary that `b` is of a class type with an appropriate `observe()` member function defined.

The two remaining probabilistic operators are:

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

| ------------ | ------------ |
| `?` Query    | `!` Get      |

See below for the behaviour of these operators.

### Overloading

The action of standard operators is defined by overloads in Birch code, declared using the `operator` statement.

The precedence of operators is always the same. It cannot be manipulated by Birch code.

Only the standard operators may be overloaded. All other operators have in-built behaviour as described above. It is still possible to manipulate the behaviour of some operators that cannot be overloaded. For example, the behaviour of the assignment operator `<-` can be manipulated by declaring assignments and conversions in class declarations, as described above.

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

    a:A! <- c();
    while (a?) {
      f(a!);
    }
    
It is the postfix `?` operator that triggers the continuation of the fiber execution, which proceeds until the next yield point. Repeated use of the postfix `!` operator between calls of the postfix `?` operator will retrieve the last yield value, without further execution.

Within the body of a fier, the `yield` statement is used to pause execution. This yields execution to the caller, along with the given value.

The fiber terminates when execution reaches the end of the body, if ever. When terminating, it does not yield a value. To terminate the execution of a fiber before reaching the end of the body, use an empty `return;` statement.

# The Birch Standard Library

The Birch Standard Library provides the basic functionality needed by most Birch programs, such as standard math, linear algebra, probability distributions, and I/O. It is documented separately.

# The Birch Eclipse Plugin

The Birch Eclipse Plugin provides a syntax-highlighting text editor for the Eclipse IDE. 

An Eclipse installation site has yet to be set up for the plugin. In the meantime, it is available in a separate `Birch.Eclipse` repository.

To install the plugin, first ensure that your Eclipse environment has the appropriate components. Use *Help > Install New Software...*, and install:

  * Eclipse Java Development Tools
  * Eclipse Plug-in Development Environment

Import the `Birch.Eclipse` project:

  1. *File > Import... > Git > Projects from Git > Clone URI*.
  2. Enter the URI: https://github.com/lawmurray/Birch.Eclipse.git.
  3. *Next*, then *Next* again, the branch *master* should be checked.
  4. Set the local directory to clone to, we suggest changing `.../git/...` to `.../workspace/...`.
  5. *Next* again.
  6. *Import existing Eclipse projects*.
  7. *Finish*.

The project should compile automatically, otherwise use `Project > Build Project`.

To install:

  1. *File > Export... > Plug-in Development > Deployable plug-ins and fragments*.
  2. Check the box against *Birch (x.y.z)*.
  3. Select *Install into host*.
