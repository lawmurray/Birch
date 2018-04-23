/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function doParent() -> Delay? {
    if (ρ.isDirichlet()) {
      return ρ;
    } else {
      return nil;
    }
  }

  function doSimulate() -> Integer {
    if (ρ.isDirichlet()) {
      return simulate_dirichlet_categorical(ρ.getDirichlet());
    } else {
      return simulate_categorical(ρ.value());
    }
  }

  function doObserve(x:Integer) -> Real {
    if (ρ.isDirichlet()) {
      return observe_dirichlet_categorical(x, ρ.getDirichlet());
    } else {
      return observe_categorical(x, ρ.value());
    }
  }

  function doCondition(x:Integer) {
    if (ρ.isDirichlet()) {
      ρ.setDirichlet(update_dirichlet_categorical(x, ρ.getDirichlet()));
    }
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Expression<Real[_]>) -> Categorical {
  m:Categorical(ρ);
  m.initialize();
  return m;
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical {
  return Categorical(Literal(ρ));
}
