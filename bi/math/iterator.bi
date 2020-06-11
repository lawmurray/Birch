/**
 * Iterator reduction.
 *
 * - x: Iterator.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Real!, init:Real, op:\(Real, Real) -> Real) -> Real {
  auto result <- init;
  while x? {
    result <- op(result, x!);
  }
  return result;
}

/**
 * Iterator reduction.
 *
 * - x: Iterator.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Integer!, init:Integer, op:\(Integer, Integer) -> Integer) -> Integer {
  auto result <- init;
  while x? {
    result <- op(result, x!);
  }
  return result;
}

/**
 * Iterator reduction.
 *
 * - x: Iterator.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Boolean!, init:Boolean, op:\(Boolean, Boolean) -> Boolean) -> Boolean {
  auto result <- init;
  while x? {
    result <- op(result, x!);
  }
  return result;
}

/**
 * Sum over range.
 */
function sum(x:Real!) -> Real {
  return reduce(x, 0.0, \(x:Real, y:Real) -> Real { return x + y; });
}

/**
 * Sum over range.
 */
function sum(x:Integer!) -> Integer {
  return reduce(x, 0, \(x:Integer, y:Integer) -> Integer { return x + y; });
}

/**
 * Sum over range.
 */
function sum(x:Boolean!) -> Boolean {
  return reduce(x, false, \(x:Boolean, y:Boolean) -> Boolean { return x + y; });
}

/**
 * Maximum over range.
 */
function max(x:Real!) -> Real {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Real, y:Real) -> Real { return max(x, y); });
}

/**
 * Maximum over range.
 */
function max(x:Integer!) -> Integer {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Integer, y:Integer) -> Integer { return max(x, y); });
}

/**
 * Maximum over range.
 */
function max(x:Boolean!) -> Boolean {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Boolean, y:Boolean) -> Boolean { return max(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Real!) -> Real {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Real, y:Real) -> Real { return min(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Integer!) -> Integer {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Integer, y:Integer) -> Integer { return min(x, y); });
}

/**
 * Minimum over range.
 */
function min(x:Boolean!) -> Boolean {
  x?;
  auto init <- x!;
  return reduce(x, init, \(x:Boolean, y:Boolean) -> Boolean { return min(x, y); });
}

/**
 * Create a new iterator that wraps around another, but upper bounds its trip
 * count.
 *
 * - x: Iterator.
 * - l: Maximum trip count.
 */
fiber limit(x:Real!, l:Integer) -> Real {
  assert l >= 0;
  auto i <- 1;
  while i <= l && x? {
    yield x!;
    i <- i + 1;
  }
}

/**
 * Create a new iterator that wraps around another, but upper bounds its trip
 * count.
 *
 * - x: Iterator.
 * - l: Maximum trip count.
 */
fiber limit(x:Integer!, l:Integer) -> Integer {
  assert l >= 0;
  auto i <- 1;
  while i <= l && x? {
    yield x!;
    i <- i + 1;
  }
}

/**
 * Create a new iterator that wraps around another, but upper bounds its trip
 * count.
 *
 * - x: Iterator.
 * - l: Maximum trip count.
 */
fiber limit(x:Boolean!, l:Integer) -> Boolean {
  assert l >= 0;
  auto i <- 1;
  while i <= l && x? {
    yield x!;
    i <- i + 1;
  }
}
