/**
 * Compare two empirical distributions for the purposes of tests.
 *
 * - X1: First empirical distribution.
 * - X2: Second empirical distribution.
 *
 * Return: The computed test distance and suggested pass threshold.
 */
function compare(X1:Real[_,_], X2:Real[_,_]) -> (Real, Real) {
  assert rows(X1) == rows(X2);
  assert columns(X1) == columns(X2);
  
  R:Integer <- rows(X1);
  C:Integer <- columns(X1);
  
  /* project onto a random unit vector for univariate comparison */
  u:Real[_] <- simulate_uniform_unit_vector(C);  
  x1:Real[_] <- X1*u;
  x2:Real[_] <- X2*u;
  
  /* normalise onto the interva [0,1] */
  mn:Real <- min(min(x1), min(x2));
  mx:Real <- max(max(x1), max(x2));
  x1 <- (x1 - mn)/(mx - mn);
  x2 <- (x2 - mn)/(mx - mn);
  
  /* compute distance and suggested pass threshold */
  δ:Real <- wasserstein(x1, x2);
  ε:Real <- sqrt(1.0/R);
  
  return (δ, ε);
}
