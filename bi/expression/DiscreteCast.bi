/**
 * Lazy cast to discrete type.
 */
final class DiscreteCast<From,To>(x:From) <
    ScalarUnaryExpression<From,To>(x) {
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
    single!.grad(0.0);
  }
}

/**
 * Lazy cast.
 */
function Integer<From>(x:Expression<From>) -> Expression<Integer> {
  if x.isConstant() {
    return box(Integer(x.value()));
  } else {
    m:DiscreteCast<Expression<From>,Integer>(x);
    return m;
  }
}

/**
 * Lazy cast.
 */
function Boolean<From>(x:Expression<From>) -> Expression<Boolean> {
  if x.isConstant() {
    return box(Integer(x.value()));
  } else {
    m:DiscreteCast<Expression<From>,Integer>(x);
    return m;
  }
}
