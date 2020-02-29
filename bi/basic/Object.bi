/**
 * Root class of all other classes.
 */
abstract class Object {
  /**
   * Class name. This is the name of the most specific type of the object.
   */
  final function getClassName() -> String;

  /**
   * Finalize. This is called immediately before destruction and deallocation
   * of an object. Object resurrection is supported: if the function creates
   * a new reference to this object, destruction and deallocation will not
   * proceed.
   */
  function finalize();

  /**
   * Read.
   */
  function read(buffer:Buffer?) {
    if buffer? {
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
    if buffer? {
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
  return x.get() == y.get();
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
