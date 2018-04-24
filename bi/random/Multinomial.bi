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

  function graft() {
    if (ρ.isDirichlet()) {
      m:DelayDirichletCategorical(this, n.value(), ρ.getDirichlet());
      m.graft();
      delay <- m;
    } else {
      m:DelayCategorical(this, n.value(), ρ.value());
      m.graft();
      delay <- m;
    }
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) -> Multinomial {
  m:Multinomial(n, ρ);
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
