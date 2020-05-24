/**
 * Lazy `asin`.
 */
final class Asin<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  override function computeValue(x:Argument) -> Value {
    return asin(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d/sqrt(1.0 - x*x);
  }
}

/**
 * Lazy `asin`.
 */
function asin(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(asin(x.value()));
  } else {
    m:Asin<Real,Real>(x);
    return m;
  }
}
