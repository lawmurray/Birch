/**
 * Sum of a vector.
 */
function sum(x:Real[_]) -> Real {
  if (length(x) > 0) {
    r:Real <- x[1];
    for (n:Integer in 2..length(x)) {
      r <- r + x[n];
    }
    return r;
  } else {
    return 0.0;
  }
}

/**
 * Sum of a vector.
 */
function sum(x:Integer[_]) -> Integer {
  if (length(x) > 0) {
    r:Integer <- x[1];
    for (n:Integer in 2..length(x)) {
      r <- r + x[n];
    }
    return r;
  } else {
    return 0;
  }
}

/**
 * Maximum of a vector.
 */
function max(x:Real[_]) -> Real {
  assert length(x) > 0;
  r:Real <- x[1];
  for (n:Integer in 2..length(x)) {
    r <- max(r, x[n]);
  }
  return r;
}

/**
 * Maximum of a vector.
 */
function max(x:Integer[_]) -> Integer {
  assert length(x) > 0;
  r:Integer <- x[1];
  for (n:Integer in 2..length(x)) {
    r <- max(r, x[n]);
  }
  return r;
}

/**
 * Minimum of a vector.
 */
function min(x:Real[_]) -> Real {
  assert length(x) > 0;
  r:Real <- x[1];
  for (n:Integer in 2..length(x)) {
    r <- min(r, x[n]);
  }
  return r;
}

/**
 * Minimum of a vector.
 */
function min(x:Integer[_]) -> Integer {
  assert length(x) > 0;
  r:Integer <- x[1];
  for (n:Integer in 2..length(x)) {
    r <- min(r, x[n]);
  }
  return r;
}

/**
 * Inclusive prefix sum.
 */
function inclusive_prefix_sum(x:Real[_]) -> Real[_] {
  assert length(x) > 0;
  r:Real[length(x)];  
  r[1] <- x[1];
  for (n:Integer in 2..length(x)) {
    r[n] <- r[n - 1] + x[n];
  }
  return r;
}

/**
 * Inclusive prefix sum.
 */
function exclusive_prefix_sum(x:Real[_]) -> Real[_] {
  assert length(x) > 0;
  r:Real[length(x)];
  r[1] <- 0.0;
  for (n:Integer in 2..length(x)) {
    r[n] <- r[n - 1] + x[n - 1];
  }
  return r;
}

/**
 * Inclusive prefix sum.
 */
function adjacent_difference(x:Real[_]) -> Real[_] {
  assert length(x) > 0;
  r:Real[length(x)];
  r[1] <- x[1];
  for (n:Integer in 2..length(x)) {
    r[n] <- x[n] - x[n - 1];
  }
  return r;
}
