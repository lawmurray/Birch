/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Expression<Real>) ->
    Expression<Real[_]> {
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value()/right.value()));
  } else {
    return diagonal(1.0/right, left.rows())*left;
  }
}

/**
 * Lazy multivariate divide.
 */
operator (left:Real[_]/right:Expression<Real>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left/right.value()));
  } else {
    return Boxed(left)/right;
  }
}

/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Real) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()/right));
  } else {
    return left/Boxed(right);
  }
}
