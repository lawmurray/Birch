/**
 * Generic reader.
 */
class Reader {
  /**
   * Read a Boolean.
   *
   * name: Name of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readBoolean(name:String) -> Boolean?;

  /**
   * Read an integer.
   *
   * name: Name of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readInteger(name:String) -> Integer?;

  /**
   * Read a real.
   *
   * name: Name of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readReal(name:String) -> Real?;
  
  /**
   * Read a Boolean.
   *
   * path: Path of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readBoolean(path:[String]) -> Boolean?;

  /**
   * Read an integer.
   *
   * path: Path of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readInteger(path:[String]) -> Integer?;

  /**
   * Read a real.
   *
   * path: Path of the entry.
   *
   * Returns an optional with a value of the given entry exists and is of a
   * compatible type.
   */
  function readReal(path:[String]) -> Real?;
}
