/*
 * ed multinomial random variate.
 */
final class Multinomial(future:Integer[_]?, futureUpdate:Boolean,
    n:Expression<Integer>, ρ:Expression<Real[_]>) < Distribution<Integer[_]>(future, futureUpdate) {
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
    return simulate_multinomial(n, ρ);
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_multinomial(x, n, ρ);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:Dirichlet?;
      if (m <- ρ.graftDirichlet())? {
        delay <- DirichletMultinomial(future, futureUpdate, n, m!);
      } else {
        delay <- Multinomial(future, futureUpdate, n, ρ);
      }
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Multinomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function Multinomial(future:Integer[_]?, futureUpdate:Boolean, n:Integer,
    ρ:Real[_]) -> Multinomial {
  m:Multinomial(future, futureUpdate, n, ρ);
  return m;
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
