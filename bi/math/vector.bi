/**
 * Create vector filled with a given scalar.
 */
function vector(x:Real, length:Integer) -> Real[_] {
  z:Real[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::fill(first, last, x);
  }}
  return z;
}

/**
 * Create vector filled with a given value.
 */
function vector(x:Integer, length:Integer) -> Integer[_] {
  z:Integer[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::fill(first, last, x);
  }}
  return z;
}

/**
 * Create vector filled with a given value.
 */
function vector(x:Boolean, length:Integer) -> Boolean[_] {
  z:Boolean[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::fill(first, last, x);
  }}
  return z;
}

/**
 * Create vector filled with a sequentially incrementing values.
 *
 * - x: Initial value.
 * - length: Length.
 */
function iota(x:Real, length:Integer) -> Real[_] {
  z:Real[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::iota(first, last, x);
  }}
  return z;
}

/**
 * Create vector filled with a sequentially incrementing values.
 *
 * - x: Initial value.
 * - length: Length.
 */
function iota(x:Integer, length:Integer) -> Integer[_] {
  z:Integer[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::iota(first, last, x);
  }}
  return z;
}

/**
 * Create vector filled with a sequentially incrementing values.
 *
 * - x: Initial value.
 * - length: Length.
 */
function iota(x:Boolean, length:Integer) -> Boolean[_] {
  z:Boolean[length];
  cpp{{
  auto first = z.begin();
  auto last = first + z.rows();
  std::iota(first, last, x);
  }}
  return z;
}

/**
 * Convert single-element vector to scalar value.
 */
function scalar(x:Real[_]) -> Real {
  assert length(x) == 1;  
  return x[1];
}

/**
 * Convert single-element vector to scalar value.
 */
function scalar(x:Integer[_]) -> Integer {
  assert length(x) == 1;  
  return x[1];
}

/**
 * Convert single-element vector to scalar value.
 */
function scalar(x:Boolean[_]) -> Boolean {
  assert length(x) == 1;  
  return x[1];
}

/**
 * Length of a vector.
 */
function length(x:Object[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Real[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Integer[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Boolean[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Object?[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Real?[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Integer?[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Length of a vector.
 */
function length(x:Boolean?[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}

/**
 * Sum of a vector.
 */
function sum(x:Real[_]) -> Real {
  return reduce<Real>(x, 0.0, @(x:Real, y:Real) -> Real {
      return x + y; });
}

/**
 * Sum of a vector.
 */
function sum(x:Integer[_]) -> Integer {
  return reduce<Integer>(x, 0, @(x:Integer, y:Integer) -> Integer {
      return x + y; });
}

/**
 * Sum of a vector.
 */
function sum(x:Boolean[_]) -> Boolean {
  return reduce<Boolean>(x, false, @(x:Boolean, y:Boolean) -> Boolean {
      return x + y; });
}

/**
 * Product of a vector.
 */
function product(x:Real[_]) -> Real {
  return reduce<Real>(x, 1.0, @(x:Real, y:Real) -> Real {
      return x*y; });
}

/**
 * Product of a vector.
 */
function product(x:Integer[_]) -> Integer {
  return reduce<Integer>(x, 1, @(x:Integer, y:Integer) -> Integer {
      return x*y; });
}

/**
 * Product of a vector.
 */
function product(x:Boolean[_]) -> Boolean {
  return reduce<Boolean>(x, true, @(x:Boolean, y:Boolean) -> Boolean {
      return x*y; });
}

/**
 * Maximum of a vector.
 */
function max(x:Real[_]) -> Real {
  assert length(x) > 0;
  return reduce<Real>(x, x[1], @(x:Real, y:Real) -> Real {
      return max(x, y); });
}

/**
 * Maximum of a vector.
 */
function max(x:Integer[_]) -> Integer {
  assert length(x) > 0;
  return reduce<Integer>(x, x[1], @(x:Integer, y:Integer) -> Integer {
      return max(x, y); });
}

/**
 * Maximum of a vector.
 */
function max(x:Boolean[_]) -> Boolean {
  assert length(x) > 0;
  return reduce<Boolean>(x, x[1], @(x:Boolean, y:Boolean) -> Boolean {
      return max(x, y); });
}

/**
 * Minimum of a vector.
 */
function min(x:Real[_]) -> Real {
  assert length(x) > 0;
  return reduce<Real>(x, x[1], @(x:Real, y:Real) -> Real {
      return min(x, y); });
}

/**
 * Minimum of a vector.
 */
function min(x:Integer[_]) -> Integer {
  assert length(x) > 0;
  return reduce<Integer>(x, x[1], @(x:Integer, y:Integer) -> Integer {
      return min(x, y); });
}

/**
 * Minimum of a vector.
 */
function min(x:Boolean[_]) -> Boolean {
  assert length(x) > 0;
  return reduce<Boolean>(x, x[1], @(x:Boolean, y:Boolean) -> Boolean {
      return min(x, y); });
}

/**
 * Inclusive prefix sum of a vector.
 */
function inclusive_scan_sum(x:Real[_]) -> Real[_] {
  return inclusive_scan<Real>(x, @(x:Real, y:Real) -> Real { return x + y; });
}

/**
 * Inclusive prefix sum of a vector.
 */
function inclusive_scan_sum(x:Integer[_]) -> Integer[_] {
  return inclusive_scan<Integer>(x, @(x:Integer, y:Integer) -> Integer {
      return x + y; });
}

/**
 * Inclusive prefix sum of a vector.
 */
function inclusive_scan_sum(x:Boolean[_]) -> Boolean[_] {
  return inclusive_scan<Boolean>(x, @(x:Boolean, y:Boolean) -> Boolean {
      return x + y; });
}

/**
 * Exclusive prefix sum of a vector.
 */
function exclusive_scan_sum(x:Real[_]) -> Real[_] {
  return exclusive_scan<Real>(x, 0.0, @(x:Real, y:Real) -> Real { return x + y; });
}

/**
 * Exclusive prefix sum of a vector.
 */
function exclusive_scan_sum(x:Integer[_]) -> Integer[_] {
  return exclusive_scan<Integer>(x, 0, @(x:Integer, y:Integer) -> Integer {
      return x + y; });
}

/**
 * Exclusive prefix sum of a vector.
 */
function exclusive_scan_sum(x:Boolean[_]) -> Boolean[_] {
  return exclusive_scan<Boolean>(x, false, @(x:Boolean, y:Boolean) -> Boolean {
      return x + y; });
}
