/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function doGraft() -> Delay? {
    m:DelayDirichlet?;
    if (m <- ρ.graftDirichlet())? {
      return DelayDirichletCategorical(this, m!);
    } else {
      return DelayCategorical(this, ρ.value());
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
