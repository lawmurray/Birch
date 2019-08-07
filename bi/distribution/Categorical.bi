/**
 * Categorical distribution.
 */
final class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_categorical(ρ);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_categorical(x, ρ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m1:DelayDirichlet?;
      m2:DelayRestaurant?;
      if (m1 <- ρ.graftDirichlet())? {
        delay <- DelayDirichletCategorical(future, futureUpdate, m1!);
      } else if (m2 <- ρ.graftRestaurant())? {
        delay <- DelayRestaurantCategorical(future, futureUpdate, m2!);
      } else if force {
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
