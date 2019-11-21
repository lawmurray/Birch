/**
 * Root class of all other classes.
 */
class Object {
  /**
   * Get the class name of the object.
   */
  final function getClassName() -> String;

  /**
   * Read.
   */
  function read(buffer:Buffer?) {
    if (buffer?) {
      read(buffer!);
    }
  }

  /**
   * Read.
   */
  function read(buffer:Buffer) {
    //
  }
  
  /**
   * Write.
   */
  function write(buffer:Buffer?) {
    if (buffer?) {
      write(buffer!);
    }
  }

  /**
   * Write.
   */
  function write(buffer:Buffer) {
    buffer.set("class", getClassName());
  }
}

/**
 * Identity comparison.
 */
operator (x:Object == y:Object) -> Boolean {
  cpp{{
  return x.get().get() == y.get().get();
  }}
}

/**
 * Identity comparison.
 */
operator (x:Object != y:Object) -> Boolean {
  return !(x == y);
}

/**
 * Identity comparison.
 */
operator (x:Object? == y:Object?) -> Boolean {
  return (x? && y? && x! == y!) || (!x? && !y?); 
}

/**
 * Identity comparison.
 */
operator (x:Object? != y:Object?) -> Boolean {
  return !(x == y);
}

/**
 * Identity conversion.
 */
function Object(o:Object) -> Object {
  return o;
}
