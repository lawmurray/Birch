/**
 * Lazy `tanh`.
 */
final class Tanh(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- tanh(single.value());
  }

  override function doGet() {
    x <- tanh(single.get());
  }

  override function doPilot() {
    x <- tanh(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- tanh(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!*(1.0 + pow(tanh(single.get()), 2.0)));
  }
}

/**
 * Lazy `tanh`.
 */
function tanh(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(tanh(x.value()));
  } else {
    m:Tanh(x);
    return m;
  }
}
