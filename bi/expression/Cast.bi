/**
 * Lazy cast.
 */
final class Cast<From,To>(x:From) < ScalarUnaryExpression<From,To>(x) {
  override function doValue() {
    x <- To?(single.value())!;
  }

  override function doGet() {
    x <- To?(single.get())!;
  }

  override function doPilot() {
    x <- To?(single.pilot())!;
  }

  override function doMove(κ:Kernel) {
    x <- To?(single.move(κ))!;
  }

  override function doGrad() {
    single.grad(d!);
  }
}

/**
 * Lazy cast.
 */
function Real<From>(x:Expression<From>) -> Expression<Real> {
  if x.isConstant() {
    return box(Real(x.value()));
  } else {
    m:Cast<Expression<From>,Real>(x);
    return m;
  }
}
