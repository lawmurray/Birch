Birch uses reference counting to determine when objects should be destroyed. Some usage patterns create reference cycles that prevent object destruction. To break these cycles, weak references may be used. A weak reference is marked by placing an ampersand (`&`) immediately after a class type:

    a:A&;

Weak references do not participate in reference counting. An object is destroyed as soon it has no references to it besides weak references. As a consequence of this, it is necessary to check the validity of a weak reference before attempting to access an object through it, as that object may have been destroyed. The usage idiom is to assign the weak reference to an optional, then check whether the optional has a value, and if so, to do something with the value:

    b:A?;
    b <- a;  // assign the weak reference to the optional
    if (b?) {  // check if b has a value
      f(b!);  // if so, do something with the value of b
    }

If the weak reference still points to a valid object, this creates a new reference to that object, contained in the optional. As this is not a weak reference, the object will not be destroyed while the reference exists. If the weak reference no longer points to a valid object, as it has previously been destroyed, then the optional will have no value.

This is the only way to access an object via a weak reference. A weak reference cannot be assigned to a non-weak reference of the same type, but a non-weak reference can be assigned to a weak reference.
