/**
 * Abstract reader.
 */
class Reader {
  /**
   * Get this as an object.
   */
  function getObject() -> Reader?;

  /**
   * Get this as an array.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber getArray() -> Reader;
  
  /**
   * Get the length of an array.
   *
   * Return: An optional with a value giving the length if this is an array.
   */
  function getLength() -> Integer?;

  /**
   * Get this as a Boolean.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getBoolean() -> Boolean?;

  /**
   * Get this as an integer.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getInteger() -> Integer?;

  /**
   * Get this as a real.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getReal() -> Real?;

  /**
   * Get this as a string.
   *
   * Return: An optional with a value if this is of a compatible type.
   */
  function getString() -> String?;

  /**
   * Get this as a vector of Booleans.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanVector() -> Boolean[_]?;

  /**
   * Get this as a vector of integers.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerVector() -> Integer[_]?;

  /**
   * Get this as a vector of reals.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealVector() -> Real[_]?;

  /**
   * Get this as a matrix of Booleans.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getBooleanMatrix() -> Boolean[_,_]?;

  /**
   * Get this as a matrix of integers.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getIntegerMatrix() -> Integer[_,_]?;

  /**
   * Get this as a matrix of reals.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getRealMatrix() -> Real[_,_]?;

  /**
   * Get an object.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getObject(name:String) -> Reader?;

  /**
   * Get an array.
   *
   * - name: Name of the entry.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber getArray(name:String) -> Reader;

  /**
   * Get the length of an array.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value giving the length if the given entry
   * is an array.
   */
  function getLength(name:String) -> Integer?;

  /**
   * Get a Boolean.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getBoolean(name:String) -> Boolean?;

  /**
   * Get an integer.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getInteger(name:String) -> Integer?;

  /**
   * Get a real.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getReal(name:String) -> Real?;

  /**
   * Get a string.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getString(name:String) -> String?;

  /**
   * Get a vector of Booleans.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanVector(name:String) -> Boolean[_]?;

  /**
   * Get a vector of integers.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerVector(name:String) -> Integer[_]?;

  /**
   * Get a vector of reals.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealVector(name:String) -> Real[_]?;

  /**
   * Get a matrix of Booleans.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getBooleanMatrix(name:String) -> Boolean[_,_]?;

  /**
   * Get a matrix of integers.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getIntegerMatrix(name:String) -> Integer[_,_]?;

  /**
   * Get a matrix of reals.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getRealMatrix(name:String) -> Real[_,_]?;

  /**
   * Get an object.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getObject(path:[String]) -> Reader?;

  /**
   * Get an array.
   *
   * - path: Path of the entry.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber getArray(path:[String]) -> Reader;

  /**
   * Get the length of an array.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value giving the length if the given entry
   * is an array.
   */
  function getLength(path:[String]) -> Integer?;

  /**
   * Get a Boolean.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getBoolean(path:[String]) -> Boolean?;

  /**
   * Get an integer.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getInteger(path:[String]) -> Integer?;

  /**
   * Get a real.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getReal(path:[String]) -> Real?;

  /**
   * Get a string.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getString(path:[String]) -> String?;

  /**
   * Get a vector of Booleans.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanVector(path:[String]) -> Boolean[_]?;

  /**
   * Get a vector of integers.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerVector(path:[String]) -> Integer[_]?;

  /**
   * Get a vector of reals.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealVector(path:[String]) -> Real[_]?;

  /**
   * Get a matrix of Booleans.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getBooleanMatrix(path:[String]) -> Boolean[_,_]?;

  /**
   * Get a matrix of integers.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getIntegerMatrix(path:[String]) -> Integer[_,_]?;

  /**
   * Get a matrix of reals.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function getRealMatrix(path:[String]) -> Real[_,_]?;

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
   * Get this as an object.
   *
   * - value: The object.
   *
   * Return: The object.
   *
   * If `value?`, calls `value!.read(this)`. Returns `value`.
   */
  function get(value:Object?) -> Object? {
    if value? {
      value!.read(this);
    }
    return value;
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(name:String, value:String?) -> String? {
    return getString(name);
  }

  /**
   * Get an object.
   *
   * - name: Name of the entry.
   * - value: The object.
   *
   * Return: The object.
   *
   * If `value?`, calls `value!.read(getObject(name))`. Returns `value`.
   */
  function get(name:String, value:Object?) -> Object? {
    if value? {
      value!.read(getObject(name));
    }
    return value;
  }

  /**
   * Get a vector of Booleans.
   *
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
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
   * - name: Name of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(name:String, value:Real[_,_]?) -> Real[_,_]? {
    return getRealMatrix(name);
  }

  /**
   * Get a Boolean.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(path:[String], value:Boolean?) -> Boolean? {
    return getBoolean(path);
  }

  /**
   * Get an integer.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(path:[String], value:Integer?) -> Integer? {
    return getInteger(path);
  }

  /**
   * Get a real.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(path:[String], value:Real?) -> Real? {
    return getReal(path);
  }

  /**
   * Get a string.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function get(path:[String], value:String?) -> String? {
    return getString(path);
  }

 /**
   * Get an object.
   *
   * - path: Path of the entry.
   * - value: The object.
   *
   * Return: The object.
   *
   * If `value?`, calls `value!.read(getObject(path))`. Returns `value`.
   */
  function get(path:[String], value:Object?) -> Object? {
    if value? {
      value!.read(getObject(path));
    }
    return value;
  }

  /**
   * Get a vector of Booleans.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(path:[String], value:Boolean[_]?) -> Boolean[_]? {
    return getBooleanVector(path);
  }

  /**
   * Get a vector of integers.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(path:[String], value:Integer[_]?) -> Integer[_]? {
    return getIntegerVector(path);
  }

  /**
   * Get a vector of reals.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function get(path:[String], value:Real[_]?) -> Real[_]? {
    return getRealVector(path);
  }

  /**
   * Get a matrix of Booleans.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(path:[String], value:Boolean[_,_]?) -> Boolean[_,_]? {
    return getBooleanMatrix(path);
  }

  /**
   * Get a matrix of integers.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(path:[String], value:Integer[_,_]?) -> Integer[_,_]? {
    return getIntegerMatrix(path);
  }

  /**
   * Get a matrix of reals.
   *
   * - path: Path of the entry.
   * - value: Unused, but necessary for overload resolution.
   *
   * Return: An optional with a value if this is an array where all elements
   * are themselves arrays of the same length and compatible type.
   */
  function get(path:[String], value:Real[_,_]?) -> Real[_,_]? {
    return getRealMatrix(path);
  }
}

/**
 * Get a value or object of a prescribed type from a Reader.
 *
 * - reader: The reader.
 *
 * Return: An optional with a value if the read is successful.
 */
function get<Type>(reader:Reader) -> Type? {
  x:Type;
  y:Type?;
  y <- Type?(reader.get(x));
  return y;
}

/**
 * Get a value or object of a prescribed type from a Reader.
 *
 * - reader: The reader.
 * - name: Name of the entry.
 *
 * Return: An optional with a value if the read is successful.
 */
//function get<Type>(reader:Reader, name:String) -> Type? {
//  return get<Type>(reader.getObject(name));
//}

/**
 * Get a value or object of a prescribed type from a Reader.
 *
 * - reader: The reader.
 * - path: Path of the entry.
 *
 * Return: An optional with a value if the read is successful.
 */
//function get<Type>(reader:Reader, path:[String]) -> Type? {
//  return get<Type>(reader.getObject(path));
//}
