/**
 * Length of a vector.
 */
function length(x:Real[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Integer[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Boolean[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Object[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Real?[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Integer?[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Boolean?[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}

/**
 * Length of a vector.
 */
function length(x:Object?[_]) -> Integer {
  cpp{{
  return x_.length(0);
  }}
}
