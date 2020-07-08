/**
 * Lazy `diagonal`.
 */
final class MultivariateDiagonal(x:Expression<Real[_,_]>) <
    MultivariateUnaryExpression<Expression<Real[_,_]>,Real[_]>(x) {
  override function doRows() -> Integer {
    return min(single!.rows(), single!.columns());
  }

  override function doValue() {
    x <- diagonal(single!.value());
  }

  override function doPilot() {
    x <- diagonal(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- diagonal(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(diagonal(d!));
  }
}

/**
 * Lazy `diagonal`.
 */
function diagonal(x:Expression<Real[_,_]>) -> Expression<Real[_]> {
  if x.isConstant() {
    return box(vector(diagonal(x.value())));
  } else {
    return construct<MultivariateDiagonal>(x);
  }
}
