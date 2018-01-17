/**
 * Abstract reader.
 */
class Reader {
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
}
