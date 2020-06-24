/**
 * Lazy `lgamma`.
 */
final class MultivariateLogGamma(x:Expression<Real>, p:Expression<Integer>) <
    ScalarBinaryExpression<Expression<Real>,Expression<Integer>,Real>(x, p) {
  override function doValue() {
    x <- lgamma(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- lgamma(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- lgamma(left!.move(κ), right!.move(κ));
  }

  override function doGrad() {
    auto y <- 0.0;
    auto x <- left!.get();
    auto p <- right!.get();
    for i in 1..p {
      y <- y + digamma(x + 0.5*(1 - i));
    }
    left!.grad(d!*y);
    right!.grad(0.0);
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Expression<Real>, p:Expression<Integer>) -> Expression<Real> {
  if x.isConstant() && p.isConstant() {
    return box(lgamma(x.value(), p.value()));
  } else {
    m:MultivariateLogGamma(x, p);
    return m;
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Real, p:Expression<Integer>) -> Expression<Real> {
  if p.isConstant() {
    return box(lgamma(x, p.value()));
  } else {
    return lgamma(box(x), p);
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Expression<Real>, p:Integer) -> Expression<Real> {
  if x.isConstant() {
    return box(lgamma(x.value(), p));
  } else {
    return lgamma(x, box(p));
  }
}
