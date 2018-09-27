/**
 * Abstract writer.
 */
class Writer {
  /**
   * Set this as an object.
   *
   * Returns: a writer (this one) for modifying the new element.
   */
  function setObject() -> Writer;
  
  /**
   * Set this as an array.
   *
   * Returns: a writer (this one) for modifying the new element.
   */
  function setArray() -> Writer;

  /**
   * Set this as nil.
   */
  function setNil();

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
   * Set this as a vector of Booleans.
   *
   * - value: Value of the entry.
   */
  function setBooleanVector(value:Boolean[_]);

  /**
   * Set this as a vector of integers.
   *
   * - value: Value of the entry.
   */
  function setIntegerVector(value:Integer[_]);

  /**
   * Set this as a vector of reals.
   *
   * - value: Value of the entry.
   */
  function setRealVector(value:Real[_]);

  /**
   * Set this as matrix of Booleans.
   *
   * - value: Value of the entry.
   */
  function setBooleanMatrix(value:Boolean[_,_]);

  /**
   * Set this as a matrix of integers.
   *
   * - value: Value of the entry.
   */
  function setIntegerMatrix(value:Integer[_,_]);

  /**
   * Set this as a matrix of reals.
   *
   * - value: Value of the entry.
   */
  function setRealMatrix(value:Real[_,_]);

  /**
   * Set an object.
   *
   * - name: Name of the entry.
   *
   * Returns: a writer for modifying the new element.
   */
  function setObject(name:String) -> Writer;
  
  /**
   * Set an array.
   *
   * - name: Name of the entry.
   *
   * Returns: a writer for modifying the new element.
   */
  function setArray(name:String) -> Writer;

  /**
   * Set this as nil.
   *
   * - name: Name of the entry.
   */
  function setNil(name:String);

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
   * Set a vector of Booleans.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setBooleanVector(name:String, value:Boolean[_]);

  /**
   * Set a vector of integers.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setIntegerVector(name:String, value:Integer[_]);

  /**
   * Set a vector of reals.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setRealVector(name:String, value:Real[_]);

  /**
   * Set a matrix of Booleans.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setBooleanMatrix(name:String, value:Boolean[_,_]);

  /**
   * Set a matrix of integers.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setIntegerMatrix(name:String, value:Integer[_,_]);

  /**
   * Set a matrix of reals.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function setRealMatrix(name:String, value:Real[_,_]);
  
  /**
   * Set an object.
   *
   * - path: Path of the entry.
   *
   * Returns: a writer for modifying the new element.
   */
  function setObject(path:[String]) -> Writer;
  
  /**
   * Set an array.
   *
   * - path: Path of the entry.
   *
   * Returns: a writer for modifying the new element.
   */
  function setArray(path:[String]) -> Writer;

  /**
   * Set as nil.
   *
   * - path: Path of the entry.
   */
  function setNil(path:[String]);

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
   * Set a vector of Booleans.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setBooleanVector(path:[String], value:Boolean[_]);

  /**
   * Set a vector of integers.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setIntegerVector(path:[String], value:Integer[_]);

  /**
   * Set a vector of reals.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setRealVector(path:[String], value:Real[_]);

  /**
   * Set a matrix of Booleans.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setBooleanMatrix(path:[String], value:Boolean[_,_]);

  /**
   * Set a matrix of integers.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setIntegerMatrix(path:[String], value:Integer[_,_]);

  /**
   * Set a matrix of reals.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function setRealMatrix(path:[String], value:Real[_,_]);

  /**
   * Set this as a Boolean.
   *
   * - value: Value of the entry.
   */
  function set(value:Boolean) {
    setBoolean(value);
  }

  /**
   * Set this as an integer.
   *
   * - value: Value of the entry.
   */
  function set(value:Integer) {
    setInteger(value);
  }

  /**
   * Set this as a real.
   *
   * - value: Value of the entry.
   */
  function set(value:Real) {
    setReal(value);
  }

  /**
   * Set this as a string.
   *
   * - value: Value of the entry.
   */
  function set(value:String) {
    setString(value);
  }
  
  /**
   * Write as an object.
   *
   * - o: The object.
   *
   * Calls `o.write(this)`.
   */
  function set(o:Object) {
    o.write(this);
  }

  /**
   * Set this as a vector of Booleans.
   *
   * - value: Value of the entry.
   */
  function set(value:Boolean[_]) {
    setBooleanVector(value);
  }

  /**
   * Set this as a vector of integers.
   *
   * - value: Value of the entry.
   */
  function set(value:Integer[_]) {
    setIntegerVector(value);
  }

  /**
   * Set this as a vector of reals.
   *
   * - value: Value of the entry.
   */
  function set(value:Real[_]) {
    setRealVector(value);
  }

  /**
   * Set this as matrix of Booleans.
   *
   * - value: Value of the entry.
   */
  function set(value:Boolean[_,_]) {
    setBooleanMatrix(value);
  }

  /**
   * Set this as a matrix of integers.
   *
   * - value: Value of the entry.
   */
  function set(value:Integer[_,_]) {
    setIntegerMatrix(value);
  }

  /**
   * Set this as a matrix of reals.
   *
   * - value: Value of the entry.
   */
  function set(value:Real[_,_]) {
    setRealMatrix(value);
  }

  /**
   * Set a Boolean.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Boolean) {
    setBoolean(name, value);
  }

  /**
   * Set an integer.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Integer) {
    setInteger(name, value);
  }

  /**
   * Set a real.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Real) {
    setReal(name, value);
  }

  /**
   * Set a string.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:String) {
    setString(name, value);
  }

  /**
   * Set a vector of Booleans.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Boolean[_]) {
    setBooleanVector(name, value);
  }

  /**
   * Set a vector of integers.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Integer[_]) {
    setIntegerVector(name, value);
  }

  /**
   * Set a vector of reals.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Real[_]) {
    setRealVector(name, value);
  }

  /**
   * Set a matrix of Booleans.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Boolean[_,_]) {
    setBooleanMatrix(name, value);
  }

  /**
   * Set a matrix of integers.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Integer[_,_]) {
    setIntegerMatrix(name, value);
  }

  /**
   * Set a matrix of reals.
   *
   * - name: Name of the entry.
   * - value: Value of the entry.
   */
  function set(name:String, value:Real[_,_]) {
    setRealMatrix(name, value);
  }
  
  /**
   * Set a Boolean.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Boolean) {
    setBoolean(path, value);
  }

  /**
   * Set an integer.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Integer) {
    setInteger(path, value);
  }

  /**
   * Set a real.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Real) {
    setReal(path, value);
  }

  /**
   * Set a string.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:String) {
    setString(path, value);
  }

  /**
   * Set a vector of Booleans.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Boolean[_]) {
    setBooleanVector(path, value);
  }

  /**
   * Set a vector of integers.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Integer[_]) {
    setIntegerVector(path, value);
  }

  /**
   * Set a vector of reals.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Real[_]) {
    setRealVector(path, value);
  }

  /**
   * Set a matrix of Booleans.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Boolean[_,_]) {
    setBooleanMatrix(path, value);
  }

  /**
   * Set a matrix of integers.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Integer[_,_]) {
    setIntegerMatrix(path, value);
  }

  /**
   * Set a matrix of reals.
   *
   * - path: Path of the entry.
   * - value: Value of the entry.
   */
  function set(path:[String], value:Real[_,_]) {
    setRealMatrix(path, value);
  }

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
