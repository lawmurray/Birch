/*
 * ed multinomial random variate.
 */
final class Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) <
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
    prune();
    m:Dirichlet?;
    if (m <- ρ.graftDirichlet())? {
      return DirichletMultinomial(n, m!);
    } else {
      return this;
    }
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
