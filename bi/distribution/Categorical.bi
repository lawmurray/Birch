/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:DelayDirichlet?;
      m2:DelayRestaurant?;
      if (m1 <- ρ.graftDirichlet())? {
        delay <- DelayDirichletCategorical(x, m1!);
      } else if (m2 <- ρ.graftRestaurant())? {
        delay <- DelayRestaurantCategorical(x, m2!);
      } else {
        delay <- DelayCategorical(x, ρ);
      }
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayCategorical(x, ρ);
    }
    return DelayValue<Integer>?(delay);
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Expression<Real[_]>) -> Categorical {
  m:Categorical(ρ);
  return m;
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical {
  return Categorical(Boxed(ρ));
}
