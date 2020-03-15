/*
 * Grafted negative binomial distribution.
 */
final class GraftedNegativeBinomial(k:Expression<Integer>,
    ρ:Expression<Real>) < NegativeBinomial(k, ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }
}

function GraftedNegativeBinomial(k:Expression<Integer>,
    ρ:Expression<Real>) -> GraftedNegativeBinomial {
  m:GraftedNegativeBinomial(k, ρ);
  return m;
}
