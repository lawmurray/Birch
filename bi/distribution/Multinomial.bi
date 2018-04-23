/**
 * Multinomial distribution.
 */
class Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) < Random<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Expression<Integer>;

  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]>;

  function doParent() -> Delay? {
    if (ρ.isDirichlet()) {
      return ρ;
    } else {
      return nil;
    }
  }

  function doSimulate() -> Integer[_] {
    if (ρ.isDirichlet()) {
      return simulate_dirichlet_multinomial(n.value(), ρ.getDirichlet());
    } else {
      return simulate_multinomial(n.value(), ρ.value());
    }
  }

  function doObserve(x:Integer[_]) -> Real {
    if (ρ.isDirichlet()) {
      return observe_dirichlet_multinomial(x, n.value(), ρ.getDirichlet());
    } else {
      return observe_multinomial(x, n.value(), ρ.value());
    }
  }

  function doCondition(x:Integer[_]) {
    if (ρ.isDirichlet()) {
      ρ.setDirichlet(update_dirichlet_multinomial(x, n.value(), ρ.getDirichlet()));
    }
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) -> Multinomial {
  m:Multinomial(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Real[_]) -> Multinomial {
  return Multinomial(n, Literal(ρ));
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Expression<Real[_]>) -> Multinomial {
  return Multinomial(Literal(n), ρ);
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Real[_]) -> Multinomial {
  return Multinomial(Literal(n), Literal(ρ));
}
