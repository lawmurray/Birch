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
   * Get this as a Boolean.
   */
  function getBoolean() -> Boolean?;

  /**
   * Get this as an integer.
   */
  function getInteger() -> Integer?;

  /**
   * Get this as a real.
   */
  function getReal() -> Real?;

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
   * path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getReal(path:[String]) -> Real?;

  /**
   * Get a string.
   *
   * path: Path of the entry.
   *
   * Return: An optional with a value if the given entry exists and is of a
   * compatible type.
   */
  function getString(path:[String]) -> String?;
}
