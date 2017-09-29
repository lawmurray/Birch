/**
 * Output stream.
 */
class OutputStream {
  /**
   * File handle.
   */
  file:File?;

  /**
   * Print string.
   */
  function print(value:String) {
    assert file?;
    cpp{{
    ::fprintf(file_.get(), "%s", value_.c_str());
    }}
  }

  /**
   * Print value.
   */
  function print(value:Boolean) {
    if (value) {
      print("true");
    } else {
      print("false");
    }
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
  function print(x:Boolean[_]) {
    for (i:Integer in 1..length(x)) {
      if (i != 1) {
        print(" ");
      }
      print(x[i]);
    }
  }

  /**
   * Print vector.
   */
  function print(x:Integer[_]) {
    for (i:Integer in 1..length(x)) {
      if (i != 1) {
        print(" ");
      }
      print(x[i]);
    }
  }

  /**
   * Print vector.
   */
  function print(x:Real[_]) {
    for (i:Integer in 1..length(x)) {
      if (i != 1) {
        print(" ");
      }
      print(x[i]);
    }
  }

  /**
   * Print matrix.
   */
  function print(X:Boolean[_,_]) {
    for (i:Integer in 1..rows(X)) {
      for (j:Integer in 1..columns(X)) {
        if (j != 1) {
          print(" ");
        }
        print(X[i,j]);
      }
      print("\n");
    }
  }

  /**
   * Print matrix.
   */
  function print(X:Integer[_,_]) {
    for (i:Integer in 1..rows(X)) {
      for (j:Integer in 1..columns(X)) {
        if (j != 1) {
          print(" ");
        }
        print(X[i,j]);
      }
      print("\n");
    }
  }

  /**
   * Print matrix.
   */
  function print(X:Real[_,_]) {
    for (i:Integer in 1..rows(X)) {
      for (j:Integer in 1..columns(X)) {
        if (j != 1) {
          print(" ");
        }
        print(X[i,j]);
      }
      print("\n");
    }
  }
}

/**
 * Constructor for output stream.
 */
function OutputStream(file:File) -> OutputStream {
  o:OutputStream;
  o.file <- file;
  return o;
}
