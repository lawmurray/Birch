/**
 * Multinomial distribution.
 */
class Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) <
    Distribution<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function rows() -> Integer {
    return ρ.rows();
  }

  function simulate() -> Integer[_] {
    return simulate_multinomial(n.value(), ρ.value());
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_multinomial(x, n.value(), ρ.value());
  }

  function graft() -> Distribution<Integer[_]> {
    if !hasValue() {
      prune();
      m:Dirichlet?;
      r:Distribution<Integer[_]>?;
    
      /* match a template */
      if (m <- ρ.graftDirichlet())? {
        r <- DirichletMultinomial(n, m!);
      }
    
      /* finalize, and if not valid, use default template */
      if !r? || !r!.graftFinalize() {
        r <- GraftedMultinomial(n, ρ);
        r!.graftFinalize();
      }
      return r!;
    } else {
      return this;
    }
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Multinomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) ->
    Multinomial {
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
