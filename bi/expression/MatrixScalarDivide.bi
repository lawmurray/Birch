/**
 * Lazy matrix divide.
 */
operator (y:Expression<Real[_,_]>/z:Expression<Real>) ->
    Expression<Real[_,_]> {
  return (1.0/z)*y;
}

/**
 * Lazy matrix divide.
 */
operator (y:Real[_,_]/z:Expression<Real>) -> Expression<Real[_,_]> {
  return (1.0/z)*box(y);
}

/**
 * Lazy matrix divide.
 */
operator (y:Expression<Real[_,_]>/z:Real) -> Expression<Real[_,_]> {
  return box(1.0/z)*y;
}
