/*
 * Grafted Bernoulli distribution.
 */
final class GraftedBernoulli(ρ:Expression<Real>) < Bernoulli(ρ) {
  function graft() -> Distribution<Boolean> {
    prune();
    return this;
  }
}

function GraftedBernoulli(ρ:Expression<Real>) -> GraftedBernoulli {
  m:GraftedBernoulli(ρ);
  return m;
}
