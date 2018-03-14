/**
 * Number of rows of a matrix.
 */
function rows(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X_.length(0);
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X_.length(0);
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X_.length(0);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X_.length(1);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X_.length(1);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X_.length(1);
  }}
}

/**
 * Create matrix filled with a given scalar value.
 */
function matrix(x:Real, rows:Integer, columns:Integer) -> Real[_,_] {
  Z:Real[rows,columns];
  cpp{{
  std::fill(Z_.begin(), Z_.end(), x_);
  }}
  return Z;
}

/**
 * Create matrix filled with a given scalar value.
 */
function matrix(x:Integer, rows:Integer, columns:Integer) -> Integer[_,_] {
  Z:Integer[rows,columns];
  cpp{{
  std::fill(Z_.begin(), Z_.end(), x_);
  }}
  return Z;
}

/**
 * Create matrix filled with a given scalar value.
 */
function matrix(x:Boolean, rows:Integer, columns:Integer) -> Boolean[_,_] {
  Z:Boolean[rows,columns];
  cpp{{
  std::fill(Z_.begin(), Z_.end(), x_);
  }}
  return Z;
}

/**
 * Create identity matrix.
 *
 * - length: Number of rows/columns.
 */
function identity(length:Integer) -> Real[_,_] {
  Z:Real[_,_] <- matrix(0.0, length, length);
  for (i:Integer in 1..length) {
    Z[i,i] <- 1.0;
  }
  return Z;
}

/**
 * Convert single-element matrix to scalar value.
 */
function scalar(X:Real[_,_]) -> Real {
  assert rows(X) == 1;  
  assert columns(X) == 1;  
  return X[1,1];
}

/**
 * Convert single-element matrix to scalar value.
 */
function scalar(X:Integer[_,_]) -> Integer {
  assert rows(X) == 1;  
  assert columns(X) == 1;  
  return X[1,1];
}

/**
 * Convert single-element matrix to scalar value.
 */
function scalar(X:Boolean[_,_]) -> Boolean {
  assert rows(X) == 1;  
  assert columns(X) == 1;  
  return X[1,1];
}
