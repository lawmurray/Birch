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
  function open(path:String, mode:String) {
    assert !(file?);
    file <- fopen(path, mode);
  }

  /**
   * Open file with default mode.
   *
   *   - path: Path.
   */
  function open(path:String) {
    open(path, "r");
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
  function' eof() -> Boolean {
    assert file?;
    cpp{{
    return ::feof(file_.get());
    }}
  }

  /**
   * Read integer.
   */
  function' readInteger() -> Integer {
    assert file?;
    cpp{{
    long long int x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(file_.get(), "%lld", &x);
    return x;
    }}
  }

  /**
   * Read real.
   */
  function' readReal() -> Real {
    assert file?;
    cpp{{
    double x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(file_.get(), "%lf", &x);
    return x;
    }}
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
