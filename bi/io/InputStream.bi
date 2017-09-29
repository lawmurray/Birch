/**
 * Input stream.
 */
class InputStream {
  /**
   * File handle.
   */
  file:File;

  /**
   * Read integer.
   */
  function readInteger() -> Integer {
    cpp{{
    long long int x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(file_, "%lld", &x);
    return x;
    }}
  }

  /**
   * Read real.
   */
  function readReal() -> Real {
    cpp{{
    double x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(file_, "%lf", &x);
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
