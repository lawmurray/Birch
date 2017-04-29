/**
 * Vectors
 * -------
 */
import math.scalar;

/**
 * Length of a vector.
 */
function length(x:Real[_]) -> Integer64 {
  cpp{{
  return x.length(0);
  }}
}
