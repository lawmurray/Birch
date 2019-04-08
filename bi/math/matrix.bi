/**
 * Number of rows of a matrix.
 */
function rows(X:Object[_,_]) -> Integer64 {
  cpp{{
  return X.length(0);
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X.length(0);
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X.length(0);
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X.length(0);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Object[_,_]) -> Integer64 {
  cpp{{
  return X.length(1);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X.length(1);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X.length(1);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X.length(1);
  }}
}

/**
 * Create matrix filled with a given scalar value.
 *
 * - x: The value.
 * - rows: Number of rows.
 * - columns: Number of columns.
 */
function matrix(x:Real, rows:Integer, columns:Integer) -> Real[_,_] {
  Z:Real[rows,columns];
  cpp{{
  auto first = Z.begin();
  auto last = first + Z.size();
  std::fill(first, last, x);
  }}
  return Z;
}

/**
 * Create matrix filled with a given scalar value.
 *
 * - x: The value.
 * - rows: Number of rows.
 * - columns: Number of columns.
 */
function matrix(x:Integer, rows:Integer, columns:Integer) -> Integer[_,_] {
  Z:Integer[rows,columns];
  cpp{{
  auto first = Z.begin();
  auto last = first + Z.size();
  std::fill(first, last, x);
  }}
  return Z;
}

/**
 * Create matrix filled with a given scalar value.
 *
 * - x: The value.
 * - rows: Number of rows.
 * - columns: Number of columns.
 */
function matrix(x:Boolean, rows:Integer, columns:Integer) -> Boolean[_,_] {
  Z:Boolean[rows,columns];
  cpp{{
  auto first = Z.begin();
  auto last = first + Z.size();
  std::fill(first, last, x);
  }}
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Real, length:Integer) -> Real[_,_] {
  Z:Real[_,_] <- matrix(0.0, length, length);
  for (i:Integer in 1..length) {
    Z[i,i] <- x;
  }
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Integer, length:Integer) -> Integer[_,_] {
  Z:Integer[_,_] <- matrix(0, length, length);
  for (i:Integer in 1..length) {
    Z[i,i] <- x;
  }
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Boolean, length:Integer) -> Boolean[_,_] {
  Z:Boolean[_,_] <- matrix(false, length, length);
  for (i:Integer in 1..length) {
    Z[i,i] <- x;
  }
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given vector.
 *
 * - x: The vector.
 */
function diagonal(x:Real[_]) -> Real[_,_] {
  auto n <- length(x);
  auto Z <- matrix(0.0, n, n);
  for (i:Integer in 1..n) {
    Z[i,i] <- x[i];
  }
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given vector.
 *
 * - x: The vector.
 */
function diagonal(x:Integer[_]) -> Integer[_,_] {
  auto n <- length(x);
  auto Z <- matrix(0, n, n);
  for (i:Integer in 1..n) {
    Z[i,i] <- x[i];
  }
  return Z;
}

/**
 * Create diagonal matrix, filling the diagonal with a given vector.
 *
 * - x: The vector.
 */
function diagonal(x:Boolean[_]) -> Boolean[_,_] {
  auto n <- length(x);
  auto Z <- matrix(false, n, n);
  for (i:Integer in 1..n) {
    Z[i,i] <- x[i];
  }
  return Z;
}

/**
 * Create identity matrix.
 *
 * - length: Number of rows/columns.
 */
function identity(length:Integer) -> Real[_,_] {
  return diagonal(1.0, length);
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

/**
 * Convert vector to matrix with single row.
 */
function row(x:Real[_]) -> Real[_,_] {
  y:Real[1,length(x)];
  y[1,1..rows(y)] <- x;
  return y;
}

/**
 * Convert vector to matrix with single row.
 */
function row(x:Integer[_]) -> Integer[_,_] {
  y:Integer[1,length(x)];
  y[1,1..rows(y)] <- x;
  return y;
}

/**
 * Convert vector to matrix with single row.
 */
function row(x:Boolean[_]) -> Boolean[_,_] {
  y:Boolean[1,length(x)];
  y[1,1..rows(y)] <- x;
  return y;
}

/**
 * Convert vector to matrix with single column.
 */
function column(x:Real[_]) -> Real[_,_] {
  y:Real[length(x),1];
  y[1..rows(y),1] <- x;
  return y;
}

/**
 * Convert vector to matrix with single column.
 */
function column(x:Integer[_]) -> Integer[_,_] {
  y:Integer[length(x),1];
  y[1..rows(y),1] <- x;
  return y;
}

/**
 * Convert vector to matrix with single column.
 */
function column(x:Boolean[_]) -> Boolean[_,_] {
  y:Boolean[length(x),1];
  y[1..rows(y),1] <- x;
  return y;
}
