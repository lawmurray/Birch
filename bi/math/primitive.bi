/**
 * Reduction.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Real[_], init:Real, op:@(Real, Real) -> Real) -> Real {
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // return std::reduce(first, last, init, op);
  // ^ C++17
  return std::accumulate(first, last, init, op);
  }}
}

/**
 * Reduction.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Integer[_], init:Integer,
    op:@(Integer, Integer) -> Integer) -> Integer {
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // return std::reduce(first, last, init, op);
  // ^ C++17
  return std::accumulate(first, last, init, op);
  }}
}

/**
 * Reduction.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce(x:Boolean[_], init:Boolean,
    op:@(Boolean, Boolean) -> Boolean) -> Boolean {
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // return std::reduce(first, last, init, op);
  // ^ C++17
  return std::accumulate(first, last, init, op);
  }}
}

/**
 * Inclusive scan.
 *
 * - x: Vector.
 * - op: Operator.
 */
function inclusive_scan(x:Real[_], op:@(Real, Real) -> Real) -> Real[_] {
  y:Real[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // std::inclusive_scan(first, last, y.begin(), op);
  // ^ C++17
  std::partial_sum(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Inclusive scan.
 *
 * - x: Vector.
 * - op: Operator.
 */
function inclusive_scan(x:Integer[_], op:@(Integer, Integer) -> Integer) ->
    Integer[_] {
  y:Integer[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // std::inclusive_scan(first, last, y.begin(), op);
  // ^ C++17
  std::partial_sum(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Inclusive scan.
 *
 * - x: Vector.
 * - op: Operator.
 */
function inclusive_scan(x:Boolean[_], op:@(Boolean, Boolean) -> Boolean) ->
    Boolean[_] {
  y:Boolean[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  // std::inclusive_scan(first, last, y.begin(), op);
  // ^ C++17
  std::partial_sum(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Exclusive scan.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function exclusive_scan(x:Real[_], init:Real,
    op:@(Real, Real) -> Real) -> Real[_] {
  assert length(x) > 0;
  y:Real[length(x)];
  //cpp{{
  // auto first = x.begin();
  // auto last = first + x.size();
  // std::exclusive_scan(first, last, y.begin(), init, op);
  // ^ C++17
  //}}
  y[1] <- init;
  for (n:Integer in 2..length(x)) {
    y[n] <- y[n - 1] + x[n - 1];
  }
  return y;
}

/**
 * Exclusive scan.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function exclusive_scan(x:Integer[_], init:Integer,
    op:@(Integer, Integer) -> Integer) -> Integer[_] {
  assert length(x) > 0;
  y:Integer[length(x)];
  //cpp{{
  // auto first = x.begin();
  // auto last = first + x.size();
  // std::exclusive_scan(first, last, y.begin(), init, op);
  // ^ C++17
  //}}
  y[1] <- init;
  for (n:Integer in 2..length(x)) {
    y[n] <- y[n - 1] + x[n - 1];
  }
  return y;
}

/**
 * Exclusive scan.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function exclusive_scan(x:Boolean[_], init:Boolean,
    op:@(Boolean, Boolean) -> Boolean) -> Boolean[_] {
  assert length(x) > 0;
  y:Boolean[length(x)];
  //cpp{{
  // auto first = x.begin();
  // auto last = first + x.size();
  // std::exclusive_scan(first, last, y.begin(), init, op);
  // ^ C++17
  //}}
  y[1] <- init;
  for (n:Integer in 2..length(x)) {
    y[n] <- y[n - 1] + x[n - 1];
  }
  return y;
}

/**
 * Adjacent difference.
 *
 * - x: Vector.
 * - op: Operator.
 */
function adjacent_difference(x:Real[_],
    op:@(Real, Real) -> Real) -> Real[_] {
  y:Real[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  std::adjacent_difference(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Adjacent difference.
 *
 * - x: Vector.
 * - op: Operator.
 */
function adjacent_difference(x:Integer[_],
    op:@(Integer, Integer) -> Integer) -> Integer[_] {
  y:Integer[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  std::adjacent_difference(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Adjacent difference.
 *
 * - x: Vector.
 * - op: Operator.
 */
function adjacent_difference(x:Boolean[_],
    op:@(Boolean, Boolean) -> Boolean) -> Boolean[_] {
  y:Boolean[length(x)];
  cpp{{
  auto first = x.begin();
  auto last = first + x.size();
  std::adjacent_difference(first, last, y.begin(), op);
  }}
  return y;
}

/**
 * Sort.
 *
 * - x: Vector.
 */
function sort(x:Real[_]) -> Real[_] {
  y:Real[_] <- x;
  cpp{{
  auto first = y.begin();
  auto last = first + y.size();
  std::sort(first, last);
  }}
  return y;
}

/**
 * Sort.
 *
 * - x: Vector.
 */
function sort(x:Integer[_]) -> Integer[_] {
  y:Integer[_] <- x;
  cpp{{
  auto first = y.begin();
  auto last = first + y.size();
  std::sort(first, last);
  }}
  return y;
}

/**
 * Sort.
 *
 * - x: Vector.
 */
function sort(x:Boolean[_]) -> Boolean[_] {
  y:Boolean[_] <- x;
  cpp{{
  auto first = y.begin();
  auto last = first + y.size();
  std::sort(first, last);
  }}
  return y;
}
