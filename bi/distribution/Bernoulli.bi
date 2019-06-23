/**
 * Bernoulli distribution.
 */
final class Bernoulli(ρ:Expression<Real>) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function simulateForward() -> Boolean {
    assert !delay?;
    return simulate_bernoulli(ρ);
  }

  function logpdfForward(x:Boolean) -> Real {
    assert !delay?;
    return logpdf_bernoulli(x, ρ);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta())? {
        delay <- DelayBetaBernoulli(future, futureUpdate, m!);
      } else {
        delay <- DelayBernoulli(future, futureUpdate, ρ);
      }
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Bernoulli");
      buffer.set("ρ", ρ.value());
    }
  }
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Expression<Real>) -> Bernoulli {
  m:Bernoulli(ρ);
  return m;
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  return Bernoulli(Boxed(ρ));
}
