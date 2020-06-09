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
    return box(left)/right;
  }
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Real) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()/right));
  } else {
    return left/box(right);
  }
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<LLT>/right:Expression<Real>) ->
    Expression<Real[_,_]> {
  return matrix(left)/right;
}

/**
 * Lazy matrix divide.
 */
operator (left:LLT/right:Expression<Real>) -> Expression<Real[_,_]> {
  return matrix(left)/right;
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<LLT>/right:Real) -> Expression<Real[_,_]> {
  return matrix(left)/right;
}
