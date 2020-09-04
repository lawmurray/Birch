/**
 * Reduction.
 *
 * - h: Fiber handle.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce<Type>(h:Real!, init:Type, op:\(Type, Type) -> Type) -> Type {
  auto result <- init;
  while h? {
    result <- op(result, h!);
  }
  return result;
}

/**
 * Sum reduction.
 */
function sum<Type>(h:Type!) -> Type {
  h?;
  return reduce(h, h!, \(x:Real, y:Real) -> Real { return x + y; });
}

/**
 * Product reduction.
 */
function product<Type>(h:Type!) -> Type {
  h?;
  return reduce(h, h!, \(x:Real, y:Real) -> Real { return x*y; });
}

/**
 * Maximum reduction.
 */
function max<Type>(h:Type!) -> Type {
  h?;
  return reduce(h, h!, \(x:Real, y:Real) -> Real { return max(x, y); });
}

/**
 * Minimum reduction.
 */
function min<Type>(h:Type!) -> Type {
  h?;
  return reduce(h, h!, \(x:Type, y:Type) -> Type { return min(x, y); });
}

/**
 * Create a new iterator that wraps around another, but upper bounds its
 * number of yields count.
 *
 * - h: Fiber handle.
 * - l: Maximum number of yields.
 */
fiber limit<Type>(h:Type!, l:Integer) -> Type {
  assert l >= 0;
  auto i <- 1;
  while i <= l && h? {
    yield h!;
    i <- i + 1;
  }
}
