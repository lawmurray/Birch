/*
 * Grafted delta distribution.
 */
class GraftedDelta(μ:Expression<Integer>) < Delta(μ) {
  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }
}

function GraftedDelta(μ:Expression<Integer>) -> GraftedDelta {
  m:GraftedDelta(μ);
  return m;
}
