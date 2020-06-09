/**
 * Lazy `log`.
 */
final class Log(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- log(single.value());
  }

  override function doGet() {
    x <- log(single.get());
  }

  override function doPilot() {
    x <- log(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- log(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!/single.get());
  }
}

/**
 * Lazy `log`.
 */
function log(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(log(x.value()));
  } else {
    m:Log(x);
    return m;
  }
}
