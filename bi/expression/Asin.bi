/**
 * Lazy `asin`.
 */
final class Asin(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- asin(single!.value());
  }

  override function doPilot() {
    x <- asin(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- asin(single!.move(κ));
  }

  override function doGrad() {
    auto x <- single!.get();
    single!.grad(d!/sqrt(1.0 - x*x));
  }
}

/**
 * Lazy `asin`.
 */
function asin(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(asin(x.value()));
  } else {
    return construct<Asin>(x);
  }
}
