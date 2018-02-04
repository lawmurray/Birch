## Read-only references

A class type may be followed by an apostrophe (prime) symbol (`'`) to denote it read-only:

    a:A';

For a reference that is both weak and read-only, the prime precedes the ampersand:

    a:A'&;

With a read-only reference to an object, only read-only functions and fibers may be called on that object. These are marked in the class definition by using a prime immediately after the `function` or `fiber` keyword.
