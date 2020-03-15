/*
 * Grafted binomial distribution.
 */
final class GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) <
    Binomial(n, ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }
  
  function graftDiscrete() -> Discrete? {
    prune();
    return this;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    prune();
    return this;
  }
}

function GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) ->
    GraftedBinomial {
  m:GraftedBinomial(n, ρ);
  return m;
}
