/**
 * Categorical distribution.
 */
final class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      m1:DelayDirichlet?;
      m2:DelayRestaurant?;
      if (m1 <- ρ.graftDirichlet(child))? {
        delay <- DelayDirichletCategorical(future, futureUpdate, m1!);
      } else if (m2 <- ρ.graftRestaurant(child))? {
        delay <- DelayRestaurantCategorical(future, futureUpdate, m2!);
      } else {
        delay <- DelayCategorical(future, futureUpdate, ρ);
      }
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
