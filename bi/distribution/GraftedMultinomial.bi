/*
 * Grafted multinomial distribution.
 */
final class GraftedMultinomial(n:Expression<Integer>,
    ρ:Expression<Real[_]>) < Multinomial(n, ρ) {
  function graft() -> Distribution<Integer[_]> {
    prune();
    return this;
  }
}

function GraftedMultinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) ->
    GraftedMultinomial {
  m:GraftedMultinomial(n, ρ);
  return m;
}
