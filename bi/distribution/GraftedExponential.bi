/*
 * Grafted exponential distribution.
 */
final class GraftedExponential(λ:Expression<Real>) < Exponential(λ) {
  function graft() -> Distribution<Real> {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    λ.value();
    return true;
  }
}

function GraftedExponential(λ:Expression<Real>) -> GraftedExponential {
  m:GraftedExponential(λ);
  return m;
}
