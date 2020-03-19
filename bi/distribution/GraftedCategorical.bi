/*
 * Grafted categorical distribution.
 */
final class GraftedCategorical(ρ:Expression<Real[_]>) < Categorical(ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    ρ.value();
    return true;
  }
}

function GraftedCategorical(ρ:Expression<Real[_]>) -> GraftedCategorical {
  m:GraftedCategorical(ρ);
  return m;
}
