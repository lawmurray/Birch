/**
 * Lazy negation.
 */
final class Negate<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function doValue(x:Argument) -> Value {
    return -x;
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
