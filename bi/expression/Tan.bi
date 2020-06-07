/**
 * Lazy `tan`.
 */
final class Tan(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- tan(single.value());
  }

  override function doPilot() {
    x <- tan(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- tan(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!*(1.0 + pow(tan(single.get()), 2.0)));
  }
}

/**
 * Lazy `tan`.
 */
function tan(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(tan(x.value()));
  } else {
    m:Tan(x);
    return m;
  }
}
