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
  function' putBoolean(name:String, value:Boolean);

  /**
   * Write an integer.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function' putInteger(name:String, value:Integer);

  /**
   * Write a real.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function' putReal(name:String, value:Real);
  
  /**
   * Write a Boolean.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function' putBoolean(path:[String], value:Boolean);

  /**
   * Write an integer.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function' putInteger(path:[String], value:Integer);

  /**
   * Write a real.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function' putReal(path:[String], value:Real);
}
