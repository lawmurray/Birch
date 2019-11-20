/**
 * Number of rows of a matrix.
 */
function rows(X:Object[_,_]) -> Integer64 {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Object[_,_]) -> Integer64 {
  cpp{{
  return X.cols();
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X.cols();
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X.cols();
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(X:Boolean[_,_]) -> Integer64 {
  cpp{{
  return X.cols();
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
  std::fill(Z.begin(), Z.end(), x);
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
  std::fill(Z.begin(), Z.end(), x);
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
  std::fill(Z.begin(), Z.end(), x);
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
  for i in 1..length {
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
  for i in 1..length {
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
  for i in 1..length {
    Z[i,i] <- x;
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
 * Convert matrix to a vector, concatenating rows.
 */
function vector(X:Boolean[_,_]) -> Boolean[_] {
  auto rows <- global.rows(X);
  auto cols <- global.columns(X);
  x:Boolean[rows*cols];
  for i in 1..rows {
    x[(i - 1)*cols + 1 .. i*cols] <- X[i,1..cols];
  }
  return x;
}

/**
 * Convert matrix to a vector, concatenating rows.
 */
function vector(X:Real[_,_]) -> Real[_] {
  auto rows <- global.rows(X);
  auto cols <- global.columns(X);
  x:Real[rows*cols];
  for i in 1..rows {
    x[(i - 1)*cols + 1 .. i*cols] <- X[i,1..cols];
  }
  return x;
}

/**
 * Convert matrix to a vector, concatenating rows.
 */
function vector(X:Integer[_,_]) -> Integer[_] {
  auto rows <- global.rows(X);
  auto cols <- global.columns(X);
  x:Integer[rows*cols];
  for i in 1..rows {
    x[(i - 1)*cols + 1 .. i*cols] <- X[i,1..cols];
  }
  return x;
}

/**
 * Convert matrix to matrix (identity operation).
 */
function matrix(X:Boolean[_,_], rows:Integer, cols:Integer) -> Boolean[_,_] {
  assert global.rows(X) == rows;
  assert global.columns(X) == cols;
  return X;
}

/**
 * Convert matrix to matrix (identity operation).
 */
function matrix(X:Real[_,_], rows:Integer, cols:Integer) -> Real[_,_] {
  assert global.rows(X) == rows;
  assert global.columns(X) == cols;
  return X;
}

/**
 * Convert matrix to matrix (identity operation).
 */
function matrix(X:Integer[_,_], rows:Integer, cols:Integer) -> Integer[_,_] {
  assert global.rows(X) == rows;
  assert global.columns(X) == cols;
  return X;
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

/**
 * Convert matrix to String.
 */
function String(X:Real[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- String(X[i,j]);
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      buf << value;
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}

/**
 * Convert matrix to String.
 */
function String(X:Integer[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- String(X[i,j]);
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      buf << value;
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}

/**
 * Convert matrix to String.
 */
function String(X:Boolean[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- String(X[i,j]);
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      buf << value;
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}
