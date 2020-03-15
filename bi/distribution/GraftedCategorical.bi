/*
 * Grafted categorical distribution.
 */
final class GraftedCategorical(ρ:Expression<Real[_]>) < Categorical(ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }
}

function GraftedCategorical(ρ:Expression<Real[_]>) -> GraftedCategorical {
  m:GraftedCategorical(ρ);
  return m;
}
