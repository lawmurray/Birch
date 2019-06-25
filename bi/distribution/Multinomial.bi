/**
 * Multinomial distribution.
 */
final class Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) < Distribution<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function valueForward() -> Integer[_] {
    assert !delay?;
    return simulate_multinomial(n, ρ);
  }

  function observeForward(x:Integer[_]) -> Real {
    assert !delay?;
    return logpdf_multinomial(x, n, ρ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayDirichlet?;
      if (m <- ρ.graftDirichlet())? {
        delay <- DelayDirichletMultinomial(future, futureUpdate, n, m!);
      } else if force {
        delay <- DelayMultinomial(future, futureUpdate, n, ρ);
      }
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Multinomial");
      buffer.set("n", n.value());
      buffer.set("ρ", ρ.value());
    }
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
