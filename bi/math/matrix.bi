/**
 * Matrices
 * --------
 */
import math.scalar;

/**
 * Number of rows of a matrix.
 */
function rows(x:Real[_,_]) -> Integer64 {
  cpp{{
  return x.length(0);
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns(x:Real[_,_]) -> Integer64 {
  cpp{{
  return x.length(1);
  }}
}
