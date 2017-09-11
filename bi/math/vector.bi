import math.scalar;

/**
 * Length of a vector.
 */
function length(x:Real[_]) -> Integer64 {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Integer[_]) -> Integer64 {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Convert single-element vector to scalar.
 */
function scalar(x:Real[_]) -> Real {
  assert length(x) == 1;  
  return x[1];
}

/**
 * Create vector filled with a given scalar.
 */
function vector(x:Real, length:Integer) -> Real[_] {
  z:Real[length];
  for (i:Integer in 1..length) {
    z[i] <- x;
  }
  return z;
}
