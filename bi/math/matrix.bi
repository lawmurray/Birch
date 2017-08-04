/**
 * Matrices
 * --------
 */
import math.scalar;

/**
 * Number of rows of a matrix.
 */
function rows(X:Real[_,_]) -> Integer64 {
  cpp{{
  return X_.length(0);
  }}
}
function rows(X:Integer[_,_]) -> Integer64 {
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
function columns(X:Integer[_,_]) -> Integer64 {
  cpp{{
  return X_.length(1);
  }}
}

/**
 * Convert single-element matrix to scalar.
 */
function scalar(X:Real[_,_]) -> Real {
  assert rows(X) == 1 && columns(X) == 1;
  
  return X[1,1];
}

/**
 * Create matrix filled with a given scalar.
 */
function matrix(x:Real, rows:Integer, columns:Integer) -> Real[_,_] {
  Z:Real[rows,columns];
  i:Integer;
  j:Integer;
  for (i in 1..rows) {
    for (j in 1..columns) {
      Z[i,j] <- x;
    }
  }
  return Z;
}

/**
 * Create identity matrix.
 */
function identity(rows:Integer, columns:Integer) -> Real[_,_] {
  Z:Real[rows,columns];
  i:Integer;
  j:Integer;
  for (i in 1..rows) {
    for (j in 1..columns) {
      if (i == j) {
        Z[i,j] <- 1.0;
      } else {
        Z[i,j] <- 0.0;
      }
    }
  }
  return Z;
}
