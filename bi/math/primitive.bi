/**
 * Unary map.
 *
 * - x: Operand.
 * - f: Operator.
 *
 * Applies `f` to each element of `X`.
 */
function for_each<Type>(x:Type[_], f:\(Type)) {
  for i in 1..length(x) {
    f(x[i]);
  }
}

/**
 * Unary map.
 *
 * - X: Operand.
 * - f: Operator.
 *
 * Applies `f` to each element of `X`.
 */
function for_each<Type>(X:Type[_,_], f:\(Type)) {
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      f(X[i,j]);
    }
  }
}

/**
 * Binary map.
 *
 * - x: First operand.
 * - y: Second operand.
 * - f: Operator.
 */
function for_each<Type1,Type2>(x:Type1[_], y:Type2[_],
    f:\(Type1, Type2)) {
  assert length(x) == length(y);
  for i in 1..length(x) {
    f(x[i], y[i]);
  }
}

/**
 * Binary map.
 *
 * - X: First operand.
 * - Y: Second operand.
 * - f: Operator.
 */
function for_each<Type1,Type2>(X:Type1[_,_], Y:Type2[_,_],
    f:\(Type1, Type2)) {
  assert rows(X) == rows(Y);
  assert columns(X) == columns(Y);
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      f(X[i,j], Y[i,j]);
    }
  }
}

/**
 * Ternary map.
 *
 * - x: First operand.
 * - y: Second operand.
 * - z: Third operand.
 * - f: Operator.
 */
function for_each<Type1,Type2,Type3>(x:Type1[_], y:Type2[_],
    z:Type3[_], f:\(Type1, Type2, Type3)) {
  assert length(x) == length(y);
  assert length(y) == length(z);
  for i in 1..length(x) {
    f(x[i], y[i], z[i]);
  }
}

/**
 * Ternary map.
 *
 * - X: First operand.
 * - Y: Second operand.
 * - Z: Third operand.
 * - f: Operator.
 */
function for_each<Type1,Type2,Type3>(X:Type1[_,_], Y:Type2[_,_],
    Z:Type3[_,_], f:\(Type1, Type2, Type3)) {
  assert rows(X) == rows(Y);
  assert rows(Y) == rows(Z);
  assert columns(X) == columns(Y);
  assert columns(Y) == columns(Z);
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      f(X[i,j], Y[i,j], Z[i,j]);
    }
  }
}

/**
 * Unary transformation.
 *
 * - x: Operand.
 * - f: Operator.
 */
function transform<Type1,Result>(x:Type1[_], f:\(Type1) -> Result) ->
    Result[_] {
  // in C++17 can use std::transform
  y:Result[length(x)];
  for i in 1..length(x) {
    y[i] <- f(x[i]);
  }
  return y;
}

/**
 * Unary transformation.
 *
 * - X: Operand.
 * - f: Operator.
 */
function transform<Type,Result>(X:Type[_,_], f:\(Type) -> Result) ->
    Result[_,_] {
  // in C++17 can use std::transform
  Y:Result[rows(X),columns(X)];
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      Y[i,j] <- f(X[i,j]);
    }
  }
  return Y;
}

/**
 * Binary transformation.
 *
 * - x: First operand.
 * - y: Second operand.
 * - f: Operator.
 */
function transform<Type1,Type2,Result>(x:Type1[_], y:Type2[_],
    f:\(Type1, Type1) -> Result) -> Result[_] {
  assert length(x) == length(y);
  z:Result[length(x)];
  for i in 1..length(x) {
    z[i] <- f(x[i], y[i]);
  }
  return y;
}

/**
 * Binary transformation.
 *
 * - X: First operand.
 * - Y: Second operand.
 * - f: Operator.
 */
function transform<Type1,Type2,Result>(X:Type1[_,_], Y:Type2[_,_],
    f:\(Type1, Type2) -> Result) -> Result[_,_] {
  assert rows(X) == rows(Y);
  assert columns(X) == columns(Y);
  Z:Result[rows(X),columns(X)];
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      Z[i,j] <- f(X[i,j], Y[i,j]);
    }
  }
  return Y;
}

/**
 * Ternary transformation.
 *
 * - x: First operand.
 * - y: Second operand.
 * - z: Third operand.
 * - f: Operator.
 */
function transform<Type1,Type2,Type3,Result>(x:Type1[_], y:Type2[_],
    z:Type3[_], f:\(Type1, Type2, Type3) -> Result) -> Result[_] {
  assert length(x) == length(y);
  assert length(y) == length(z);
  a:Result[length(x)];
  for i in 1..length(x) {
    a[i] <- f(x[i], y[i], z[i]);
  }
  return a;
}

/**
 * Ternary transformation.
 *
 * - X: First operand.
 * - Y: Second operand.
 * - Z: Third operand.
 * - f: Operator.
 */
function transform<Type1,Type2,Type3,Result>(X:Type1[_,_], Y:Type2[_,_],
    Z:Type3[_,_], f:\(Type1, Type2, Type3) -> Result) -> Result[_,_] {
  assert rows(X) == rows(Y);
  assert rows(Y) == rows(Z);
  assert columns(X) == columns(Y);
  assert columns(Y) == columns(Z);
  A:Result[rows(X),columns(X)];
  for i in 1..rows(X) {
    for j in 1..columns(X) {
      A[i,j] <- f(X[i,j], Y[i,j], Z[i,j]);
    }
  }
  return A;
}

/**
 * Reduction.
 *
 * - x: Vector.
 * - init: Initial value.
 * - op: Operator.
 */
function reduce<Type>(x:Type[_], init:Type,
    op:\(Type, Type) -> Type) -> Type {
  result:Type;
  cpp{{
  x.pin();
  // result = return std::reduce(x.begin(), x.end(), init, op);
  // ^ C++17
  result = std::accumulate(x.begin(), x.end(), init, op);
  x.unpin();
  return result;
  }}
}

/**
 * Unary transformation and reduction.
 *
 * - x: First operand.
 * - init: Initial value.
 * - op1: Reduction operator.
 * - op2: Transformation operator.
 */
function transform_reduce<Type>(x:Type[_], init:Type,
    op1:\(Type, Type) -> Type, op2:\(Type) -> Type) -> Type {
  auto y <- init;
  for n in 1..length(x) {
    y <- op1(y, op2(x[n]));
  }
  return y;
}

/**
 * Binary transformation and reduction.
 *
 * - x: First operand.
 * - y: Second operand.
 * - init: Initial value.
 * - op1: Reduction operator.
 * - op2: Transformation operator.
 */
function transform_reduce<Type>(x:Type[_], y:Type[_], init:Type,
    op1:\(Type, Type) -> Type, op2:\(Type, Type) -> Type) -> Type {
  assert length(x) == length(y);
  auto z <- init;
  for n in 1..length(x) {
    z <- op1(z, op2(x[n], y[n]));
  }
  return z;
}

/**
 * Ternary transformation and reduction.
 *
 * - x: First operand.
 * - y: Second operand.
 * - z: Third operand.
 * - init: Initial value.
 * - op1: Reduction operator.
 * - op2: Transformation operator.
 */
function transform_reduce<Type>(x:Type[_], y:Type[_], z:Type[_],
    init:Type, op1:\(Type, Type) -> Type,
    op2:\(Type, Type, Type) -> Type) -> Type {
  assert length(x) == length(y);
  assert length(y) == length(z);
  auto a <- init;
  for n in 1..length(x) {
    a <- op1(a, op2(x[n], y[n], z[n]));
  }
  return a;
}

/**
 * Inclusive scan.
 *
 * - x: Vector.
 * - op: Operator.
 */
function inclusive_scan<Type>(x:Type[_], op:\(Type, Type) -> Type) -> Type[_] {
  y:Type[length(x)];
  cpp{{
  x.pin();
  // std::inclusive_scan(x.begin(), x.end(), y.begin(), op);
  // ^ C++17
  std::partial_sum(x.begin(), x.end(), y.begin(), op);
  x.unpin();
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
function exclusive_scan<Type>(x:Type[_], init:Type,
    op:\(Type, Type) -> Type) -> Type[_] {
  assert length(x) > 0;
  y:Type[length(x)];
  //cpp{{
  // std::exclusive_scan(x.begin(), x.end(), y.begin(), init, op);
  // ^ C++17
  //}}
  y[1] <- init;
  for i in 2..length(x) {
    y[i] <- y[i - 1] + x[i - 1];
  }
  return y;
}

/**
 * Adjacent difference.
 *
 * - x: Vector.
 * - op: Operator.
 */
function adjacent_difference<Type>(x:Type[_],
    op:\(Type, Type) -> Type) -> Type[_] {
  y:Type[length(x)];
  cpp{{
  x.pin();
  std::adjacent_difference(x.begin(), x.end(), y.begin(), op);
  x.unpin();
  }}
  return y;
}

/**
 * Sort a vector.
 *
 * - x: Vector.
 *
 * Returns: A vector giving the elements of `x` in ascending order.
 */
function sort<Type>(x:Type[_]) -> Type[_] {
  auto y <- x;
  cpp{{
  y.pinWrite();
  std::sort(y.begin(), y.end());
  y.unpin();
  }}
  return y;
}

/**
 * Sort a vector.
 *
 * - x: Vector.
 *
 * Returns: A vector giving the indices of elements in `x` in ascending
 * order.
 */
function sort_index<Type>(x:Type[_]) -> Integer[_] {
  auto a <- iota(1, length(x));
  cpp{{
  x.pin();
  std::sort(a.begin(), a.end(), [=](bi::type::Integer i, bi::type::Integer j) {
      return x(libbirch::make_slice(i - 1)) < x(libbirch::make_slice(j - 1));
    });
  x.unpin();
  }}
  return a;
}

/**
 * Gather.
 *
 * - a: Indices.
 * - x: Source vector.
 *
 * Returns: a vector `y` where `y[n] == x[a[n]]`.
 */
function gather<Type>(a:Integer[_], x:Type[_]) -> Type[_] {
  auto N <- length(a);
  y:Type[N];
  for n in 1..N {
    y[n] <- x[a[n]];
  }
  return y;
}

/**
 * Scatter.
 *
 * - a: Indices.
 * - x: Source vector.
 *
 * Returns: a vector `y` where `y[a[n]] == x[n]`.
 *
 * If the same index appears more than once in `a`, the result is undefined.
 */
function scatter<Type>(a:Integer[_], x:Type[_]) -> Type[_] {
  auto N <- length(a);
  y:Type[N];
  for n in 1..N {
    y[a[n]] <- x[n];
  }
  return y;
}
