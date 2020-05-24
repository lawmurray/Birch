/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Expression<Real>) ->
    Expression<Real[_,_]> {
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()/right.value()));
  } else {
    return diagonal(1.0/right, left.rows())*left;
  }
}

/**
 * Lazy matrix divide.
 */
operator (left:Real[_,_]/right:Expression<Real>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left/right.value()));
  } else {
    return Boxed(left)/right;
  }
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Real) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()/right));
  } else {
    return left/Boxed(right);
  }
}
