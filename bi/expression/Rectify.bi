/**
 * Lazy `rectify`.
 */
final class Rectify(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- rectify(single.value());
  }

  override function doGet() {
    x <- rectify(single.get());
  }

  override function doPilot() {
    x <- rectify(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- rectify(single.move(κ));
  }

  override function doGrad() {
    if x! > 0.0 {
      single.grad(1.0);
    } else {
      single.grad(0.0);
    }
  }
}

/**
 * Lazy `rectify`.
 */
function rectify(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(rectify(x.value()));
  } else {
    m:Rectify(x);
    return m;
  }
}
