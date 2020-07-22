/**
 * 1-Wasserstein distance between two univariate empirical distributions with
 * equal number of samples.
 *
 * - x1: Samples from the first distribution.
 * - x2: Samples from the second distribution.
 *
 * Return: 1-Wasserstein distance between `x1` and `x2`.
 */
function wasserstein(x1:Real[_], x2:Real[_]) -> Real {
  assert length(x1) == length(x2);
  auto N <- length(x1);
  auto y1 <- sort<Real>(x1);
  auto y2 <- sort<Real>(x2);
  return reduce<Real>(y1 - y2, 0.0, \(a:Real, b:Real) -> Real {
      return abs(a) + abs(b); })/N;
}
