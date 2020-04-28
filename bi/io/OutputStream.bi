/**
 * Output stream.
 */
class OutputStream {
  /**
   * File handle.
   */
  file:File?;

  /**
   * Open file.
   *
   *   - path: Path.
   *   - mode: Mode. 
   */
  function open(path:String, mode:Integer) {
    file <- fopen(path, mode);
  }

  /**
   * Open file with default mode.
   *
   *   - path: Path.
   */
  function open(path:String) {
    open(path, WRITE);
  }

  /**
   * Close file.
   */
  function close() {
    assert file?;
    fclose(file!);
    file <- nil;
  }
  
  /**
   * Flush.
   */
  function flush() {
    assert file?;
    fflush(file!);
  }

  /**
   * Print string.
   */
  function print(value:String) {
    assert file?;
    cpp{{
    ::fprintf(this_()->file.get(), "%s", value.c_str());
    }}
  }

  /**
   * Print value.
   */
  function print(value:Boolean) {
    print(String(value));
  }

  /**
   * Print value.
   */
  function print(value:Integer) {
    print(String(value));
  }

  /**
   * Print value.
   */
  function print(value:Real) {
    print(String(value));
  }

  /**
   * Print vector.
   */
  function print(value:Boolean[_]) {
    print(String(value));
  }

  /**
   * Print vector.
   */
  function print(value:Integer[_]) {
    print(String(value));
  }

  /**
   * Print vector.
   */
  function print(value:Real[_]) {
    print(String(value));
  }

  /**
   * Print matrix.
   */
  function print(value:Boolean[_,_]) {
    print(String(value));
  }

  /**
   * Print matrix.
   */
  function print(value:Integer[_,_]) {
    print(String(value));
  }

  /**
   * Print matrix.
   */
  function print(value:Real[_,_]) {
    print(String(value));
  }
}

/**
 * Create an output stream for an already-open file.
 */
function OutputStream(file:File) -> OutputStream {
  o:OutputStream;
  o.file <- file;
  return o;
}
