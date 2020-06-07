/**
 * Lazy `acos`.
 */
final class Acos(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- acos(single.value());
  }

  override function doPilot() {
    x <- acos(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- acos(single.move(κ));
  }

  override function doGrad() {
    auto x <- single.get();
    single.grad(-d!/sqrt(1.0 - x*x));
  }
}

/**
 * Lazy `acos`.
 */
function acos(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(acos(x.value()));
  } else {
    m:Acos(x);
    return m;
  }
}
