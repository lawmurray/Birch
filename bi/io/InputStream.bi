/**
 * Input stream.
 */
class InputStream {
  /**
   * File handle.
   */
  file:File?;

  /**
   * Read integer.
   */
  function readInteger() -> Integer {
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
  function readReal() -> Real {
    assert file?;
    cpp{{
    double x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(file_.get(), "%lf", &x);
    return x;
    }}
  }
}

/**
 * Constructor for input stream.
 */
function InputStream(file:File) -> InputStream {
  o:InputStream;
  o.file <- file;
  return o;
}
