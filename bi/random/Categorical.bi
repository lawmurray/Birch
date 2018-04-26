/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft() {
    if (ρ.isDirichlet()) {
      m:DelayDirichletCategorical(this, ρ.getDirichlet());
      m.graft();
      delay <- m;
    } else {
      m:DelayCategorical(this, ρ.value());
      m.graft();
      delay <- m;
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
