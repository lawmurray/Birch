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

  function doGraft() -> DelayValue<Integer[_]>? {
    m:DelayDirichlet?;
    if (m <- ρ.graftDirichlet())? {
      return DelayDirichletMultinomial(this, n, m!);
    } else {
      return DelayMultinomial(this, n, ρ.value());
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
  return Multinomial(n, Boxed(ρ));
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Expression<Real[_]>) -> Multinomial {
  return Multinomial(Boxed(n), ρ);
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Real[_]) -> Multinomial {
  return Multinomial(Boxed(n), Boxed(ρ));
}
