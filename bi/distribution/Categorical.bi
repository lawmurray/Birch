/**
 * Categorical distribution.
 */
final class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:DelayDirichlet?;
      m2:DelayRestaurant?;
      if (m1 <- ρ.graftDirichlet())? {
        delay <- DelayDirichletCategorical(x, m1!);
      } else if (m2 <- ρ.graftRestaurant())? {
        delay <- DelayRestaurantCategorical(x, m2!);
      } else {
        delay <- DelayCategorical(x, ρ);
      }
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Categorical");
      buffer.set("ρ", ρ.value());
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
