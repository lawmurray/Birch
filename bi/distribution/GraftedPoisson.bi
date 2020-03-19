/*
 * Grafted Poisson distribution.
 */
class GraftedPoisson(λ:Expression<Real>) < Poisson(λ) {
  function graft() -> Distribution<Integer> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftDiscrete() -> Discrete? {
    if !hasValue() {
      prune();
      graftFinalize();
      return this;
    } else {
      return nil;
    }
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
