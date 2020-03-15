/*
 * Grafted Poisson distribution.
 */
class GraftedPoisson(λ:Expression<Real>) < Poisson(λ) {
  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }

  function graftDiscrete() -> Discrete? {
    prune();
    return this;
  }
}

function GraftedPoisson(λ:Expression<Real>) -> GraftedPoisson {
  o:GraftedPoisson(λ);
  return o;
}
