/**
 * Array value.
 */
class ArrayValue < Value {
  buffers:List<MemoryBuffer>;

  function accept(writer:Writer) {
    writer.visit(this);
  }
  
  function isArray() -> Boolean {
    return true;
  }

  function size() -> Integer {
    return buffers.size();
  }

  fiber walk() -> Buffer {
    buffers.walk();
  }

  function push() -> Buffer {
    buffer:MemoryBuffer;
    buffers.pushBack(buffer);
    return buffer;
  }

  function getBooleanVector() -> Boolean[_]? {
    result:Boolean[buffers.size()];
    auto f <- buffers.walk();
    auto i <- 1;
    while f? {
      auto value <- f!.getBoolean();
      if value? {
        result[i] <- value!;
        i <- i + 1;
      } else {
        return nil;
      }
    }
    return result;
  }

  function getIntegerVector() -> Integer[_]? {
    result:Integer[buffers.size()];
    auto f <- buffers.walk();
    auto i <- 1;
    while f? {
      auto value <- f!.getInteger();
      if value? {
        result[i] <- value!;
        i <- i + 1;
      } else {
        return nil;
      }
    }
    return result;
  }

  function getRealVector() -> Real[_]? {
    result:Real[buffers.size()];
    auto f <- buffers.walk();
    auto i <- 1;
    while f? {
      auto value <- f!.getReal();
      if !value? {
        value <- f!.getInteger();
      }
      if value? {      
        result[i] <- value!;
        i <- i + 1;
      } else {
        return nil;
      }
    }
    return result;
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    nrows:Integer? <- size();
    if (nrows?) {
      auto row <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.size();
        X:Boolean[_,_];
        x:Boolean[_]?;
        if (ncols?) {
          X <- matrix(false, nrows!, ncols!);
          x <- row!.getBooleanVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.size();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getBooleanVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    nrows:Integer? <- size();
    if (nrows?) {
      auto row <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.size();
        X:Integer[_,_];
        x:Integer[_]?;
        if (ncols?) {
          X <- matrix(0, nrows!, ncols!);
          x <- row!.getIntegerVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.size();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getIntegerVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getRealMatrix() -> Real[_,_]? {
    nrows:Integer? <- size();
    if (nrows?) {
      auto row <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.size();
        X:Real[_,_];
        x:Real[_]?;
        if (ncols?) {
          X <- matrix(0.0, nrows!, ncols!);
          x <- row!.getRealVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.size();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getRealVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }
}

function ArrayValue() -> ArrayValue {
  o:ArrayValue;
  return o;
}
