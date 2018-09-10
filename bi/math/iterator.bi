/**
 * Sum over range.
 */
function sum(x:Real!) -> Real {
  return reduce(x, 0.0, @(x:Real, y:Real) -> Real { return x + y; });
}

/**
 * Sum over range.
 */
function sum(x:Integer!) -> Integer {
  return reduce(x, 0, @(x:Integer, y:Integer) -> Integer { return x + y; });
}

/**
 * Sum over range.
 */
function sum(x:Boolean!) -> Boolean {
  return reduce(x, false, @(x:Boolean, y:Boolean) -> Boolean { return x + y; });
}

/**
 * Maximum over range.
 */
function max(x:Real!) -> Real {
  x?;
  init:Real <- x!;
  return reduce(x, init, @(x:Real, y:Real) -> Real { return max(x, y); });
}

/**
 * Maximum over range.
 */
function max(x:Integer!) -> Integer {
  x?;
  init:Integer <- x!;
  return reduce(x, init, @(x:Integer, y:Integer) -> Integer { return max(x, y); });
}

/**
 * Maximum over range.
 */
function max(x:Boolean!) -> Boolean {
  x?;
  init:Boolean <- x!;
  return reduce(x, init, @(x:Boolean, y:Boolean) -> Boolean { return max(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Real!) -> Real {
  x?;
  init:Real <- x!;
  return reduce(x, init, @(x:Real, y:Real) -> Real { return min(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Integer!) -> Integer {
  x?;
  init:Integer <- x!;
  return reduce(x, init, @(x:Integer, y:Integer) -> Integer { return min(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Boolean!) -> Boolean {
  x?;
  init:Boolean <- x!;
  return reduce(x, init, @(x:Boolean, y:Boolean) -> Boolean { return min(x, y); });
}
