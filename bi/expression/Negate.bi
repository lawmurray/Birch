/**
 * Lazy negation.
 */
final class Negate<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return -x;
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return -d;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real>) -> Negate<Real,Real> {
  m:Negate<Real,Real>(x);
  return m;
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Integer>) -> Negate<Integer,Integer> {
  m:Negate<Integer,Integer>(x);
  return m;
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_]>) -> Negate<Real[_],Real[_]> {
  m:Negate<Real[_],Real[_]>(x);
  return m;
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_,_]>) -> Negate<Real[_,_],Real[_,_]> {
  m:Negate<Real[_,_],Real[_,_]>(x);
  return m;
}
