/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Distribution<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformScaledGamma?;
      m2:DelayGamma?;
      
      if (m1 <- λ.graftScaledGamma())? {
        delay <- DelayScaledGammaPoisson(x, m1!.a, m1!.x);
      }
      else if (m2 <- λ.graftGamma())? {
        delay <- DelayGammaPoisson(x, m2!);
      } else {
        delay <- DelayPoisson(x, λ);
      }
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayPoisson(x, λ);
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
