/*
 * Grafted negative binomial distribution.
 */
final class GraftedNegativeBinomial(k:Expression<Integer>,
    ρ:Expression<Real>) < NegativeBinomial(k, ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    k.value();
    ρ.value();
    return true;
  }
}

function GraftedNegativeBinomial(k:Expression<Integer>,
    ρ:Expression<Real>) -> GraftedNegativeBinomial {
  m:GraftedNegativeBinomial(k, ρ);
  return m;
}
