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
operator (-x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(-x.value());
  } else {
    m:Negate<Real,Real>(x);
    return m;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Integer>) -> Expression<Integer> {
  if x.isConstant() {
    return box(-x.value());
  } else {
    m:Negate<Integer,Integer>(x);
    return m;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_]>) -> Expression<Real[_]> {
  if x.isConstant() {
    return box(vector(-x.value()));
  } else {
    m:Negate<Real[_],Real[_]>(x);
    return m;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(-x.value()));
  } else {
    m:Negate<Real[_,_],Real[_,_]>(x);
    return m;
  }
}
