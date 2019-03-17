/**
 * Array value.
 */
class ArrayValue < Value {
  values:List<Value>;

  function getLength() -> Integer? {
    return values.size();
  }

  function getBooleanVector() -> Boolean[_]? {
    result:Boolean[values.size()];
    auto f <- values.walk();
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
    result:Integer[values.size()];
    auto f <- values.walk();
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
    result:Integer[values.size()];
    auto f <- values.walk();
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
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
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
          ncols <- row!.getLength();
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
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
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
          ncols <- row!.getLength();
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
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
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
          ncols <- row!.getLength();
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
