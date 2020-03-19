/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function simulate() -> Integer {
    return simulate_categorical(ρ.value());
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_categorical(x, ρ.value());
  }

  function cdf(x:Integer) -> Real? {
    return cdf_categorical(x, ρ.value());
  }

  function quantile(P:Real) -> Integer? {
    return quantile_categorical(P, ρ.value());
  }

  function lower() -> Integer? {
    return 1;
  }

  function upper() -> Integer? {
    return ρ.rows();
  }

  function graft() -> Distribution<Integer> {
    prune();
    m1:Dirichlet?;
    m2:Restaurant?;
    r:Distribution<Integer>?;
    
    /* match a template */
    if (m1 <- ρ.graftDirichlet())? {
      r <- DirichletCategorical(m1!);
    } else if (m2 <- ρ.graftRestaurant())? {
      r <- RestaurantCategorical(m2!);
    }
    
    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedCategorical(ρ);
      r!.graftFinalize();
    }
    return r!;
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Categorical");
    buffer.set("ρ", ρ);
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
