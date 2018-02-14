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
  fiber getArray() -> Reader!;
  
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
   * Get this as an array of Booleans.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanArray() -> Boolean[_]?;

  /**
   * Get this as an array of integers.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerArray() -> Integer[_]?;

  /**
   * Get this as an array of reals.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealArray() -> Real[_]?;

  /**
   * Get this as an array of strings.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getStringArray() -> String[_]?;

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
  fiber getArray(name:String) -> Reader!;

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
   * Get an array of Booleans.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanArray(name:String) -> Boolean[_]?;

  /**
   * Get an array of integers.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerArray(name:String) -> Integer[_]?;

  /**
   * Get an array of reals.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealArray(name:String) -> Real[_]?;

  /**
   * Get an array of strings.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getStringArray(name:String) -> String[_]?;

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
  fiber getArray(path:[String]) -> Reader!;

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
   * Get an array of Booleans.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getBooleanArray(path:[String]) -> Boolean[_]?;

  /**
   * Get an array of integers.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getIntegerArray(path:[String]) -> Integer[_]?;

  /**
   * Get an array of reals.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getRealArray(path:[String]) -> Real[_]?;

  /**
   * Get an array of strings.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if this is an array with all elements
   * of a compatible type.
   */
  function getStringArray(path:[String]) -> String[_]?;
}
