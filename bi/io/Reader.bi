/**
 * Abstract reader.
 */
class Reader {
  /**
   * This as an object.
   */
  function' get() -> Reader'?;

  /**
   * This as an array.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber' getArray() -> Reader'!;
  
  /**
   * This as a boolean.
   */
  function' getBoolean() -> Boolean?;

  /**
   * This as an integer.
   */
  function' getInteger() -> Integer?;

  /**
   * This as a real.
   */
  function' getReal() -> Real?;

  /**
   * Read an object.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' get(name:String) -> Reader'?;

  /**
   * Read an array.
   *
   * - name: Name of the entry.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber' getArray(name:String) -> Reader'!;

  /**
   * Read a Boolean.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getBoolean(name:String) -> Boolean?;

  /**
   * Read an integer.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getInteger(name:String) -> Integer?;

  /**
   * Read a real.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getReal(name:String) -> Real?;

  /**
   * Read a string.
   *
   * - name: Name of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getString(name:String) -> String?;

  /**
   * Read an object.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' get(path:[String]) -> Reader'?;

  /**
   * Read an array.
   *
   * - path: Path of the entry.
   *
   * Return: a fiber object that yields each of the elements of the array in
   * turn, or which never yields if this is an empty array or not an array
   * at all.
   */
  fiber' getArray(path:[String]) -> Reader'!;

  /**
   * Read a Boolean.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getBoolean(path:[String]) -> Boolean?;

  /**
   * Read an integer.
   *
   * - path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getInteger(path:[String]) -> Integer?;

  /**
   * Read a real.
   *
   * path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getReal(path:[String]) -> Real?;

  /**
   * Read a string.
   *
   * path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function' getString(path:[String]) -> String?;
}
