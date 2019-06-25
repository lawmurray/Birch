/**
 * Poisson distribution.
 */
final class Poisson(λ:Expression<Real>) < Distribution<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_poisson(λ);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_poisson(x, λ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformScaledGamma?;
      m2:DelayGamma?;
      
      if (m1 <- λ.graftScaledGamma())? {
        delay <- DelayScaledGammaPoisson(future, futureUpdate, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- DelayGammaPoisson(future, futureUpdate, m2!);
      } else if force {
        delay <- DelayPoisson(future, futureUpdate, λ);
      }
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayPoisson(future, futureUpdate, λ);
    }
    return DelayDiscrete?(delay);
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Poisson");
      buffer.set("λ", λ.value());
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  m:Poisson(λ);
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Boxed(λ));
}
