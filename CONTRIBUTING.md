# Contributing

Contributions are most welcome!

We love to see compelling applications of Birch. These set our direction. If you've been working hard on an interesting problem using Birch, your suggestions are valuable to us, so please let us know what you think.

There is a list of [startup projects](https://github.com/lawmurray/Birch/labels/startup%20project) that form good entry points for new contributors. As soon as you have *something* to show we can reserve a project for you. If you have another project in mind, please contact Lawrence Murray (lawrence@indii.org) to chat, who can help you shape your idea into the broader vision and design of Birch.

To make a contribution, create a pull request on the [GitHub repository](https://github.com/lawmurray/Birch). For large features, reach out to us first to make sure it will fit. Bug fixes and small features are always welcome, just go ahead and create a pull request (well, you can still talk to us first, but don't feel like you have to!). If you'd like some early feedback before your work is quite complete, go ahead and open a pull request, and just comment there that it's a work in progress. You can continue updating it until it's ready for merge.

We take code quality seriously, and are happy to help you improve yours. We will review your pull request and provide constructive feedback. We only ask that you're receptive to that. We're kind, really.

## Code style

A consistent code style is important for readability and maintainability. A good rule of thumb is that it should not be obvious to another developer that you wrote the code. Follow the lead: there is plenty of code in the standard library to follow by example. If you're not sure what to look for, here are some notes.

### For Birch code (e.g. in libraries or examples)

* Indent. Two spaces per level. Not four spaces please, because that would be two indents. Tab characters are definitely out. Yes we know there are arguments for tabs. We even know there are arguments against tabs. Sometimes we discuss driving on the left or driving on the right. It's such a struggle. Indent with two spaces, please. For the same reason we should drive on the left (there isn't one).

* Wrap lines at 78 characters, breaking after an operator or other punctuation, and indent overflow lines by four spaces instead of the usual two, e.g.

  ```
  function very_long_function_name(the_first_parameter:Real,
      the_second_parameter:Real) -> Real {
    doSomething(the_first_parameter);
    doSomethingElse(the_second_parameter);
  }
  ```

  Wrapping is not a hard rule, sometimes code is easier to read without. Use your judgement.

* Type names (including class names) use `CamelCase`.

* Member function names and member variable names use `camelCase()`.

* Global function names use `snake_case()`.

* Global variable names also use `snake_case`, but avoid those anyway. If they are meant to be constants, use `SNAKE_CASE`.

* Function parameters and local variables use `snake_case`, too.

* There is no such thing as a `Camel_Snake`.

* For variable names that correspond to a mathematical description---from a paper, say---prefer matching names to that mathematical description over these rules. For example, a variable representing a matrix may use an uppercase letter, `A`, regardless of whether it's a global, member or local variable, or a parameter. Use an underscore for subscripts, e.g. `A_x`.

* Birch supports Greek letters, so use them. Write `α` not `alpha`. Some exceptions to this: standard math functions like `gamma` and `beta` are spelled out, as the precedent is inherited, and distribution names like `Gamma` and `Beta`. The easiest way to write Greek letters is to install a Greek keyboard that you can switch to with a keyboard shortcut, or just copy-and-paste from a character map. It's not the most efficient, but it's not the bottleneck in your productivity either, and makes for easy reading later.

* Birch also supports `'`  (i.e. prime) at the end of variable names, e.g. `x'`. Useful for temporary variables, but don't overdo it.

* To improve readability, put spaces around low precedence operators, and not around high precedence operators. For example, write `a*b + c`, or `a/b - c`, not `a * b + c` or `a/b-c`.

* If a class or function should show up in the [documentation](https://docs.birch-lang.org), use a documentation comment (`/** ... */`) with the following template:

  ```
  /**
   * Do something (i.e. a brief one sentence description).
   *
   * - x: The something to do.
   *
   * Return: Did we actually do the something?
   *
   * Further details of the function can be provided here. Any Markdown can
   * be used, including math, e.g. $p(x)$, or
   * [Mermaid](https://mermaid-js.github.io) diagrams. Wrap documentation like
   * this at 78 characters.
   */
  function do_something(x:Something) -> Boolean {
    // ...
  }
  ```

### For C++ code (e.g. in driver program and LibBirch)

Much the same as for Birch code, adapted accordingly. Documentation comments are formatted for [Doxygen](https://www.doxygen.nl) instead.
