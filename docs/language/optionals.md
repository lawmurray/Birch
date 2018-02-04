## Optionals

Optional types allow variables to have a value of a particular type, or no value at all. An optional type is indicated by following any other type with `?`, like so:

    a:A?;

To check whether a variable of optional type has a value, use the postfix `?` operator, which returns `true` if there is a value and `false` if not. If there is a value, use the postfix `!` operator to retrieve it. A common usage idiom is as follows:

    if (a?) {  // check if a has a value
      f(a!);  // if so, do something with the value of a
    }

The special value of `nil` may be assigned to an optional to remove an existing value (if any):

    a <- nil;

!!! note
    In Birch, a variable of class type always has a value. In some other languages (e.g. Java), variables of class type may have a null value, and this null value is often used to denote no value. In Birch, optionals are always used where a variable may have no value. This is particularly useful when writing functions that accept arguments of class type, as there is no need to check whether those arguments actually have a value or not; they will always have a value, unless denoted as optional.
