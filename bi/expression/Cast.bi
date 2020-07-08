/**
 * Lazy cast.
 */
final class Cast<From,To>(x:From) < ScalarUnaryExpression<From,To>(x) {
  override function doValue() {
    x <- To?(single!.value())!;
  }

  override function doPilot() {
    x <- To?(single!.pilot())!;
  }

  override function doMove(κ:Kernel) {
    x <- To?(single!.move(κ))!;
  }

  override function doGrad() {
    single!.grad(d!);
  }
}

/**
 * Lazy cast.
 */
function Real(x:Expression<Integer>) -> Expression<Real> {
  if x.isConstant() {
    return box(Real(x.value()));
  } else {
    return construct<Cast<Expression<Integer>,Real>>(x);
  }
}

/**
 * Lazy cast.
 */
function Real(x:Expression<Boolean>) -> Expression<Real> {
  if x.isConstant() {
    return box(Real(x.value()));
  } else {
    return construct<Cast<Expression<Boolean>,Real>>(x);
  }
}

/**
 * Lazy cast, identity function.
 */
function Real(x:Expression<Real>) -> Expression<Real> {
  return x;
}
