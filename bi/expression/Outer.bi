/**
 * Lazy `outer`.
 */
final class Outer<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {
  override function rows() -> Integer {
    return left.rows();
  }
  
  override function columns() -> Integer {
    return right.rows();
  }
  
  override function doValue() {
    x <- outer(left.value(), right.value());
  }

  override function doGet() {
    x <- outer(left.get(), right.get());
  }

  override function doPilot() {
    x <- outer(left.pilot(), right.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- outer(left.move(κ), right.move(κ));
  }

  override function doGrad() {
    left.grad(d!*right.get());
    right.grad(transpose(d!)*left.get());
  }
}

/**
 * Lazy `outer`.
 */
function outer(left:Expression<Real[_]>, right:Expression<Real[_]>) ->
    Expression<Real[_,_]> {
  assert left.rows() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(matrix(outer(left.value(), right.value())));
  } else {
    m:Outer<Expression<Real[_]>,Expression<Real[_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy `outer`.
 */
function outer(left:Real[_], right:Expression<Real[_]>) ->
    Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(outer(left, right.value())));
  } else {
    return outer(box(left), right);
  }
}

/**
 * Lazy `outer`.
 */
function outer(left:Expression<Real[_]>, right:Real[_]) ->
    Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(outer(left.value(), right)));
  } else {
    return outer(left, box(right));
  }
}

/**
 * Lazy `outer`.
 */
function outer(single:Expression<Real[_]>) -> Expression<Real[_,_]> {
  return outer(single, single);
}
