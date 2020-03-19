/*
 * Grafted Poisson distribution.
 */
class GraftedPoisson(λ:Expression<Real>) < Poisson(λ) {
  function graft() -> Distribution<Integer> {
    prune();
    graftFinalize();
    return this;
  }

  function graftDiscrete() -> Discrete? {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    λ.value();
    return true;
  }
}

function GraftedPoisson(λ:Expression<Real>) -> GraftedPoisson {
  o:GraftedPoisson(λ);
  return o;
}
