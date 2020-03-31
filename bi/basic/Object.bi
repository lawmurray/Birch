/**
 * Root class of all other classes.
 */
abstract class Object {
  /**
   * Class name. This is the name of the most specific type of the object.
   */
  final function getClassName() -> String;

  /**
   * Read.
   */
  function read(buffer:Buffer) {
    //
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
 * Identity conversion.
 */
function Object(o:Object) -> Object {
  return o;
}
