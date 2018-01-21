/**
 * Abstract writer.
 */
class Writer {
  /**
   * Set this as an object.
   */
  function setObject();
  
  /**
   * Set this as an array.
   */
  function setArray();

  /**
   * Set this as a Boolean.
   *
   * - value: Value of the entry.
   */
  function setBoolean(value:Boolean);

  /**
   * Set this as an integer.
   *
   * - value: Value of the entry.
   */
  function setInteger(value:Integer);

  /**
   * Set this as a real.
   *
   * - value: Value of the entry.
   */
  function setReal(value:Real);

  /**
   * Set this as a string.
   *
   * - value: Value of the entry.
   */
  function setString(value:String);

  /**
   * Set an object.
   *
   * - name: Name of the entry.
   */
  function setObject(name:String);
  
  /**
   * Set an array.
   *
   * - name: Name of the entry.
   */
  function setArray(name:String);

  /**
   * Set a Boolean.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setBoolean(name:String, value:Boolean);

  /**
   * Set an integer.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setInteger(name:String, value:Integer);

  /**
   * Set a real.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setReal(name:String, value:Real);

  /**
   * Set a string.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setString(name:String, value:String);
  
  /**
   * Set an object.
   *
   * - path: Path of the entry.
   */
  function setObject(path:[String]);
  
  /**
   * Set an array.
   *
   * - path: Path of the entry.
   */
  function setArray(path:[String]);

  /**
   * Set a Boolean.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setBoolean(path:[String], value:Boolean);

  /**
   * Set an integer.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setInteger(path:[String], value:Integer);

  /**
   * Set a real.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setReal(path:[String], value:Real);

  /**
   * Set a string.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setString(path:[String], value:String);
  
  /**
   * Push a new element onto the end of an array.
   *
   * Returns: a writer for modifying the new element.
   */
  function push() -> Writer;
  
  /**
   * Flush in-memory buffers.
   */
  function flush() {
    //
  }
}
