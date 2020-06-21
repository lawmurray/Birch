/**
 * Lazy `vector`.
 */
final class MultivariateVector<Single,Value>(x:Single) <
    MultivariateUnaryExpression<Single,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }

  override function doValue() {
    x <- vector(single.value());
  }

  override function doGet() {
    x <- vector(single.get());
  }

  override function doPilot() {
    x <- vector(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- vector(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!);
  }
}

/**
 * Lazy `vector`.
 */
function vector(x:Expression<Real[_]>) -> Expression<Real[_]> {
  if x.isRandom() {
    /* Random objects are wrapped as the accumulation of gradients by element
     * requires this; see note in split() also */
    if x.isConstant() {
      return box(vector(x.value()));
    } else {
      m:MultivariateVector<Expression<Real[_]>,Real[_]>(x);
      return m;
    }
  } else {
    return x;
  }
}
