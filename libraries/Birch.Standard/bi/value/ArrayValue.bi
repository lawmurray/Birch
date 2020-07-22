/**
 * Array value.
 */
class ArrayValue < Value {
  /**
   * Elements of the array. This uses Vector rather than List or similar to
   * avoid reaching stack size limits for large data sets.
   */
  buffers:Vector<MemoryBuffer>;

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
    @buffers.walk();
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
        value <- Real?(f!.getInteger());
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
    auto nrows <- size();
    auto row <- walk();
    if row? {
      /* determine number of columns from first row */
      auto ncols <- row!.size();
      auto X <- matrix(false, nrows, ncols);
      auto x <- row!.getBooleanVector();
      if x? {
        X[1,1..ncols] <- x!;
       } else {
        return nil;
      }

      /* read in remaining rows, requiring that they have the same number of
       * columns as the first */
      auto i <- 1;
      while row? {
        ncols <- row!.size();
        if ncols == columns(X) {
          x <- row!.getBooleanVector();
          if x? {
            i <- i + 1;
            X[i,1..ncols] <- x!;
          } else {
            return nil;
          }
        } else {
          return nil;
        }
      }
      assert i == nrows;
      return X;
    }
    return nil;
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    auto nrows <- size();
    auto row <- walk();
    if row? {
      /* determine number of columns from first row */
      auto ncols <- row!.size();
      auto X <- matrix(0, nrows, ncols);
      auto x <- row!.getIntegerVector();
      if x? {
        X[1,1..ncols] <- x!;
       } else {
        return nil;
      }

      /* read in remaining rows, requiring that they have the same number of
       * columns as the first */
      auto i <- 1;
      while row? {
        ncols <- row!.size();
        if ncols == columns(X) {
          x <- row!.getIntegerVector();
          if x? {
            i <- i + 1;
            X[i,1..ncols] <- x!;
          } else {
            return nil;
          }
        } else {
          return nil;
        }
      }
      assert i == nrows;
      return X;
    }
    return nil;
  }

  function getRealMatrix() -> Real[_,_]? {
    auto nrows <- size();
    auto row <- walk();
    if row? {
      /* determine number of columns from first row */
      auto ncols <- row!.size();
      auto X <- matrix(0.0, nrows, ncols);
      auto x <- row!.getRealVector();
      if x? {
        X[1,1..ncols] <- x!;
       } else {
        return nil;
      }

      /* read in remaining rows, requiring that they have the same number of
       * columns as the first */
      auto i <- 1;
      while row? {
        ncols <- row!.size();
        if ncols == columns(X) {
          x <- row!.getRealVector();
          if x? {
            i <- i + 1;
            X[i,1..ncols] <- x!;
          } else {
            return nil;
          }
        } else {
          return nil;
        }
      }
      assert i == nrows;
      return X;
    }
    return nil;
  }
}
