/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Expression<Real>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return diagonal(1.0/right, left.rows())*left;
}

/**
 * Lazy multivariate divide.
 */
operator (left:Real[_]/right:Expression<Real>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return Boxed(left)/right;
}

/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Real) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return left/Boxed(right);
}
