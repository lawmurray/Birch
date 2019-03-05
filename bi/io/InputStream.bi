/**
 * Input stream. Supports sequential reading only, ignoring white space.
 */
class InputStream {
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
    assert !(file?);
    file <- fopen(path, mode);
  }

  /**
   * Open file with default mode.
   *
   *   - path: Path.
   */
  function open(path:String) {
    open(path, READ);
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
   * Check for end-of-file.
   */
  function eof() -> Boolean {
    assert file?;
    cpp{{
    return ::feof(self->file_.get());
    }}
  }

  /**
   * Read integer.
   */
  function scanInteger() -> Integer? {
    assert file?;
    x:Integer?;
    cpp{{
    long long int y;  // ensure fscanf gets exactly the type it expects
    auto res = ::fscanf(self->file_.get(), "%lld", &y);
    if (res == 1) {
      x_ = y;
    } 
    }}
    return x;
  }

  /**
   * Read real.
   */
  function scanReal() -> Real? {
    assert file?;
    x:Real?;
    cpp{{
    double y;  // ensure fscanf gets exactly the type it expects
    auto res = ::fscanf(self->file_.get(), "%lf", &y);
    if (res == 1) {
      x_ = y;
    } 
    }}
    return x;
  }
}

/**
 * Create an input stream from an already-open file.
 */
function InputStream(file:File) -> InputStream {
  o:InputStream;
  o.file <- file;
  return o;
}
