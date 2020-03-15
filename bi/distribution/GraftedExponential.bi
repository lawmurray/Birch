/*
 * Grafted exponential distribution.
 */
final class GraftedExponential(λ:Expression<Real>) < Exponential(λ) {
  function graft() -> Distribution<Real> {
    prune();
    return this;
  }
}

function GraftedExponential(λ:Expression<Real>) -> GraftedExponential {
  m:GraftedExponential(λ);
  return m;
}
