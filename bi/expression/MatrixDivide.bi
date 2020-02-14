/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(1.0/right, left.rows())*left;
}

/**
 * Lazy matrix divide.
 */
operator (left:Real[_,_]/right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left)/right;
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Real) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left/Boxed(right);
}
