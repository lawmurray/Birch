/*
 * Grafted multinomial distribution.
 */
final class GraftedMultinomial(n:Expression<Integer>,
    ρ:Expression<Real[_]>) < Multinomial(n, ρ) {
  function graft() -> Distribution<Integer[_]> {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    n.value();
    ρ.value();
    return true;
  }
}

function GraftedMultinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) ->
    GraftedMultinomial {
  m:GraftedMultinomial(n, ρ);
  return m;
}
