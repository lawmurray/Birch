/*
 * Grafted binomial distribution.
 */
final class GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) <
    Binomial(n, ρ) {
  function graft() -> Distribution<Integer> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftFinalize() -> Boolean {
    n.value();
    ρ.value();
    return true;
  }
}

function GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) ->
    GraftedBinomial {
  m:GraftedBinomial(n, ρ);
  return m;
}
