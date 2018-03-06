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

Member functions that do not modify the state of an object may be marked as read-only by placing a prime (`'`) immediately after the `function` keyword:

    class A {
      function' f(c:C) {
        //
      }
    }

When an object is accessed through a read-only reference, the only member functions that may be called upon it are such read-only member functions.

### Member fibers

Fiber declarations that appear within the body of a class are *member fibers*. Their behaviour is analogous to member functions. They may be similarly marked as read-only by placing a prime (`'`) immediately after the `fiber` keyword.

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

!!! note
    Initialization parameters in Birch play a similar role to initialization lists in C++.

Initialization parameters are used for simple object initialization, such as to set initial values and array sizes. They do not allow arbitrary code to be executed upon object construction. This is the role of a *constructor*. Birch does not, however, have any special language support for constructors. Instead, it is idiomatic to use *factory functions*, exploiting the fact that the same name can be used for both a function and a class in the Birch language.

A factory function is given the same name as the class it is intended to construct:

    function A(b:B, c:C) -> A {
      a:A;
      // ...
      return a;
    }

This function is treated as any other---there is nothing special about it---but it is idiomatic that such a function should return an object of the same type as its name, or of a subtype of that type. The possibility of returning a subtype makes a factory function slightly more flexible than an ordinary constructor.

For complex object construction, it can be useful to define a member function within the class that does most of the work, with the factory function simply instantiating the object, then passing its arguments to this function. It is idiomatic for such a member function to be given the name `make`.

### Assignments

!!! note
    Recall that, for basic types, assignment is by value, while for class types, assignment is by reference.

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
