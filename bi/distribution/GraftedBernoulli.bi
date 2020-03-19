/*
 * Grafted Bernoulli distribution.
 */
final class GraftedBernoulli(ρ:Expression<Real>) < Bernoulli(ρ) {
  function graft() -> Distribution<Boolean> {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    ρ.value();
    return true;
  }
}

function GraftedBernoulli(ρ:Expression<Real>) -> GraftedBernoulli {
  m:GraftedBernoulli(ρ);
  return m;
}
