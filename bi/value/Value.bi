/**
 * Abstract value.
 */
class Value {
  /**
   * Accept a generator.
   */
  function accept(gen:Generator);

  /**
   * Is this an object?
   */
  function isObject() -> Boolean {
    return false;
  }

  /**
   * Is this an array?
   */
  function isArray() -> Boolean {
    return false;
  }

  /**
   * Is this neither an object or an array?
   */
  function isScalar() -> Boolean {
    return false;
  }

  /**
   * Get a child in an object.
   */
  function getChild(name:String) -> Buffer? {
    return nil;
  }

  /**
   * Set a child in an object.
   */
  function setChild(name:String) -> Buffer {
    assert false;
  }

  /**
   * If this is an array, get its size.
   *
   * Return: An optional with a value giving the length if this is an array.
   */
  function size() -> Integer {
    assert false;
  }

  /**
   * Iterate through elements of an array.
   */
  fiber walk() -> Buffer {
    assert false;
  }

  /**
   * Push a value onto the end of an array.
   */
  function push() -> Buffer {
    assert false;
  }

  /**
   * Get this as an object.
   */
  function getObject() -> ObjectValue? {
    return nil;
  }

  /**
   * Get this as an array.
   */
  function getArray() -> ArrayValue? {
    return nil;
  }
  
  /**
   * Get this as a Boolean.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getBoolean() -> Boolean? {
    return nil;
  }

  /**
   * Get this as an integer.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getInteger() -> Integer? {
    return nil;
  }

  /**
   * Get this as a real.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getReal() -> Real? {
    return nil;
  }

  /**
   * Get this as a string.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getString() -> String? {
    return nil;
  }

  /**
   * Get this as a vector of Booleans.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanVector() -> Boolean[_]? {
    return nil;
  }

  /**
   * Get this as a vector of integers.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerVector() -> Integer[_]? {
    return nil;
  }

  /**
   * Get this as a vector of reals.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealVector() -> Real[_]? {
    return nil;
  }

  /**
   * Get this as a matrix of Booleans.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getBooleanMatrix() -> Boolean[_,_]? {
    return nil;
  }

  /**
   * Get this as a matrix of integers.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getIntegerMatrix() -> Integer[_,_]? {
    return nil;
  }

  /**
   * Get this as a matrix of reals.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getRealMatrix() -> Real[_,_]? {
    return nil;
  }
}
