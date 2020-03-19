/*
 * Grafted delta distribution.
 */
class GraftedDelta(μ:Expression<Integer>) < Delta(μ) {
  function graft() -> Distribution<Integer> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftFinalize() -> Boolean {
    μ.value();
    return true;
  }
}

function GraftedDelta(μ:Expression<Integer>) -> GraftedDelta {
  m:GraftedDelta(μ);
  return m;
}
