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

    yield b.observe(a);

Consequently, it is necessary that `b` is of a class type with an appropriate `observe()` member function defined. If the operator is used outside of a fiber, the `yield` is omitted.

The two remaining probabilistic operators are:

|               |                   |
| ------------- | ----------------- |
| `<~` Simulate | `~` Distribute as |

Like the assignment operator, these operators have no return type, and may only be used in statements, where they have the lowest, and final, precedence.

These are also syntactic sugar; `a <~ b;` means exactly:

    a <- b.simulate();

and `a ~ b;` means exactly:

    if (a.isMissing()) {
      a <- b;
    } else {
      a ~> b;
    }

### Query-Get

These are postfix unary operators used with optional and fiber types. They are of equal precedence, and of higher precedence than all other operators:

|              |              |
| ------------ | ------------ |
| `?` Query    | `!` Get      |

### Overloading

The action of standard operators is defined by overloads, declared using the `operator` statement. Only the standard operators may be overloaded. All other operators have in-built behaviour as described above.

!!! note
    It is still possible to manipulate the behaviour of some operators that cannot be overloaded. For example, the behaviour of the assignment operator `<-` can be manipulated by declaring assignments and conversions in class declarations.

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
