/**
 * Input stream.
 */
class InputStream(stream:File) {
  /**
   * Read integer.
   */
  function readInteger() -> Integer {
    cpp{{
    long long int x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(stream_, "%lld", &x);
    return x;
    }}
  }

  /**
   * Read real.
   */
  function readReal() -> Real {
    cpp{{
    double x;  // ensure fscanf gets exactly the type it expects
    ::fscanf(stream_, "%lf", &x);
    return x;
    }}
  }
}
