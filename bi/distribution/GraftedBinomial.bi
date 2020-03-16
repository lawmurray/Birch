/*
 * Grafted binomial distribution.
 */
final class GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) <
    Binomial(n, ρ) {
  function graft() -> Distribution<Integer> {
    prune();
    graftFinalize();
    return this;
  }
  
  function graftDiscrete() -> Discrete? {
    prune();
    graftFinalize();
    return this;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
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

function GraftedBinomial(n:Expression<Integer>, ρ:Expression<Real>) ->
    GraftedBinomial {
  m:GraftedBinomial(n, ρ);
  return m;
}
