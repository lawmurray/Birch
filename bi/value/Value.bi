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
   * Get a child.
   */
  function getChild(name:String) -> Buffer? {
    return nil;
  }

  /**
   * Set a child.
   */
  function setChild(name:String) -> Buffer;

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
   * If this is an array, get its length.
   *
   * Return: An optional with a value giving the length if this is an array.
   */
  function getLength() -> Integer? {
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

  /**
   * Get the length of an array.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value giving the length if the given entry
   * is an array.
   */
  function getLength(name:String) -> Integer? {
    auto child <- getChild(name);
    if child? {
      return child!.getLength();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a Boolean.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getBoolean(name:String) -> Boolean? {
    auto child <- getChild(name);
    if child? {
      return child!.getBoolean();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as an integer.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getInteger(name:String) -> Integer? {
    auto child <- getChild(name);
    if child? {
      return child!.getInteger();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a real.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getReal(name:String) -> Real? {
    auto child <- getChild(name);
    if child? {
      return child!.getReal();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a string.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getString(name:String) -> String? {
    auto child <- getChild(name);
    if child? {
      return child!.getString();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as an object.
   *
   * - name: Name of the child.
   * - value: The object into which to read.
   *
   * Return: An optional with a value if the given entry exists.
   */
  function getObject(name:String, value:Object) -> Object? {
    auto child <- getChild(name);
    if child? {
      return child!.getObject(value);
    } else {
      return nil;
    }
  }


  /**
   * Get a child as a vector of Booleans.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanVector(name:String) -> Boolean[_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getBooleanVector();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a vector of integers.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerVector(name:String) -> Integer[_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getIntegerVector();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a vector of reals.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealVector(name:String) -> Real[_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getRealVector();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a matrix of Booleans.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getBooleanMatrix(name:String) -> Boolean[_,_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getBooleanMatrix();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a matrix of integers.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getIntegerMatrix(name:String) -> Integer[_,_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getIntegerMatrix();
    } else {
      return nil;
    }
  }

  /**
   * Get a child as a matrix of reals.
   *
   * - name: Name of the child.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getRealMatrix(name:String) -> Real[_,_]? {
    auto child <- getChild(name);
    if child? {
      return child!.getRealMatrix();
    } else {
      return nil;
    }
  }
  
  /**
   * Get this as a Boolean.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function get(value:Boolean?) -> Boolean? {
    return getBoolean();
  }

  /**
   * Get this as an integer.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function get(value:Integer?) -> Integer? {
    return getInteger();
  }
  
  /**
   * Get this as a real.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function get(value:Real?) -> Real? {
    return getReal();
  }

  /**
   * Get this as a string.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function get(value:String?) -> String? {
    return getString();
  }

  /**
   * Get this as a vector of Booleans.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(value:Boolean[_]?) -> Boolean[_]? {
    return getBooleanVector();
  }

  /**
   * Get this as a vector of integers.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(value:Integer[_]?) -> Integer[_]? {
    return getIntegerVector();
  }

  /**
   * Get this as a vector of reals.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(value:Real[_]?) -> Real[_]? {
    return getRealVector();
  }

  /**
   * Get this as a matrix of Booleans.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(value:Boolean[_,_]?) -> Boolean[_,_]? {
    return getBooleanMatrix();
  }

  /**
   * Get this as a matrix of integers.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(value:Integer[_,_]?) -> Integer[_,_]? {
    return getIntegerMatrix();
  }

  /**
   * Get this as a matrix of reals.
   *
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(value:Real[_,_]?) -> Real[_,_]? {
    return getRealMatrix();
  }

  /**
   * Get a Boolean.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(name:String, value:Boolean?) -> Boolean? {
    return getBoolean(name);
  }

  /**
   * Get an integer.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(name:String, value:Integer?) -> Integer? {
    return getInteger(name);
  }

  /**
   * Get a real.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(name:String, value:Real?) -> Real? {
    return getReal(name);
  }

  /**
   * Get a string.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(name:String, value:String?) -> String? {
    return getString(name);
  }

  /**
   * Get a vector of Booleans.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(name:String, value:Boolean[_]?) -> Boolean[_]? {
    return getBooleanVector(name);
  }

  /**
   * Get a vector of integers.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(name:String, value:Integer[_]?) -> Integer[_]? {
    return getIntegerVector(name);
  }

  /**
   * Get a vector of reals.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(name:String, value:Real[_]?) -> Real[_]? {
    return getRealVector(name);
  }

  /**
   * Get a matrix of Booleans.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(name:String, value:Boolean[_,_]?) -> Boolean[_,_]? {
    return getBooleanMatrix(name);
  }

  /**
   * Get a matrix of integers.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(name:String, value:Integer[_,_]?) -> Integer[_,_]? {
    return getIntegerMatrix(name);
  }

  /**
   * Get a matrix of reals.
   *
   * - name: Name of the child.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(name:String, value:Real[_,_]?) -> Real[_,_]? {
    return getRealMatrix(name);
  }

  /**
   * Set child as an object.
   *
   * - name: Name of the child.
   */
  function setObject(name:String) {
    assert false;
  }
  
  /**
   * Set child as an array.
   *
   * - name: Name of the child.
   */
  function setArray(name:String) {
    assert false;
  }

  /**
   * Set child as nil.
   *
   * - name: Name of the child.
   */
  function setNil(name:String) {
    assert false;
  }

  /**
   * Set child as a Boolean.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setBoolean(name:String, value:Boolean?) {
    assert false;
  }

  /**
   * Set child as an integer.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setInteger(name:String, value:Integer?) {
    assert false;
  }

  /**
   * Set child as a real.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setReal(name:String, value:Real?) {
    assert false;
  }

  /**
   * Set child as a string.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setString(name:String, value:String?) {
    assert false;
  }

  /**
   * Set child as an object.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setObject(name:String, value:Object?) {
    assert false;
  }

  /**
   * Set child as a vector of Booleans.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setBooleanVector(name:String, value:Boolean[_]?) {
    assert false;
  }

  /**
   * Set child as a vector of integers.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setIntegerVector(name:String, value:Integer[_]?) {
    assert false;
  }

  /**
   * Set child as a vector of reals.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setRealVector(name:String, value:Real[_]?) {
    assert false;
  }

  /**
   * Set child as a matrix of Booleans.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setBooleanMatrix(name:String, value:Boolean[_,_]?) {
    assert false;
  }

  /**
   * Set child as a matrix of integers.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setIntegerMatrix(name:String, value:Integer[_,_]?) {
    assert false;
  }

  /**
   * Set child as a matrix of reals.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function setRealMatrix(name:String, value:Real[_,_]?) {
    assert false;
  }

  /**
   * Set a Boolean.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Boolean?) {
    setBoolean(name, value);
  }

  /**
   * Set an integer.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Integer?) {
    setInteger(name, value);
  }

  /**
   * Set a real.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Real?) {
    setReal(name, value);
  }

  /**
   * Set a string.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:String?) {
    setString(name, value);
  }

  /**
   * Set an object.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Object?) {
    setObject(name, value);
  }
  
  /**
   * Set a vector of Booleans.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Boolean[_]?) {
    setBooleanVector(name, value);
  }

  /**
   * Set a vector of integers.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Integer[_]?) {
    setIntegerVector(name, value);
  }

  /**
   * Set a vector of reals.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Real[_]?) {
    setRealVector(name, value);
  }

  /**
   * Set a matrix of Booleans.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Boolean[_,_]?) {
    setBooleanMatrix(name, value);
  }

  /**
   * Set a matrix of integers.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Integer[_,_]?) {
    setIntegerMatrix(name, value);
  }

  /**
   * Set a matrix of reals.
   *
   * - name: Name of the child.
   * - value: Value.
   */
  function set(name:String, value:Real[_,_]?) {
    setRealMatrix(name, value);
  }

  /**
   * Iterate through the elements of an array.
   *
   * Yields: Values for each element in turn.
   */
  fiber walk() -> Buffer {
    assert false;
  }

  /**
   * Iterate through the elements of an array.
   *
   * Yields: Values for each element in turn.
   */
  fiber walk(name:String) -> Buffer {
    auto child <- getChild(name);
    if child? {
      child!.walk();
    }
  }

  /**
   * Push a value onto the end of an array.
   */
  function push() -> Buffer {
    assert false;
  }
}
