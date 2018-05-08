/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft() -> DelayValue<Integer> {
    m1:DelayDirichlet?;
    m2:DelayRestaurant?;
    if (m1 <- ρ.graftDirichlet())? {
      return DelayDirichletCategorical(m1!);
    } else if (m2 <- ρ.graftRestaurant())? {
      return DelayRestaurantCategorical(m2!);
    } else {
      return DelayCategorical(ρ);
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    return DelayCategorical(ρ);
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
