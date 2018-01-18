/**
 * Abstract writer.
 */
class Writer {
  /**
   * Write a Boolean.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Boolean);

  /**
   * Write an integer.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Integer);

  /**
   * Write a real.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Real);
  
  /**
   * Write a Boolean.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Boolean);

  /**
   * Write an integer.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Integer);

  /**
   * Write a real.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Real);
}
