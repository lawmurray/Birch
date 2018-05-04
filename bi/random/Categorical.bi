/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      m1:DelayDirichlet?;
      m2:DelayRestaurant?;
      if (m1 <- ρ.graftDirichlet())? {
        return DelayDirichletCategorical(this, m1!);
      } else if (m2 <- ρ.graftRestaurant())? {
        return DelayRestaurantCategorical(this, m2!);
      } else {
        return DelayCategorical(this, ρ);
      }
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if (delay?) {
      return DelayValue<Integer>?(delay);
    } else {
      return DelayCategorical(this, ρ);
    }
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
