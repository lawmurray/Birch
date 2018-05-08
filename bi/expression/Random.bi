
/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {  
  /**
   * Value.
   */
  x:Value?;

  /**
   * Associated distribution.
   */
  dist:Distribution<Value>?;

  /**
   * Associated node on delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !this.x?;
    assert !dist?;
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !this.x?;
    assert !dist?;
    this.x <- x;
  }

  /**
   * Attach a distribution to this random variable.
   */
  function assume(dist:Distribution<Value>) {
    assert !x?;
    assert !this.dist?;
    this.dist <- dist;
  }

  /**
   * Get the value of the random variable, forcing realization if necessary.
   */
  function value() -> Value {
    if !x? {
      if !delay? {
        assert dist?;
        delay <- dist!.graft();
        delay!.x <- this;
      }
      x <- delay!.simulate();
      delay <- nil;
    }
    return x!;
  }

  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    return !x?;
  }

  function graftGaussian() -> DelayGaussian? {
    if (delay?) {
      return DelayGaussian?(delay);
    } else if (dist?) {
      return dist!.graftGaussian();
    } else {
      return nil;
    }
  }
    
  function graftBeta() -> DelayBeta? {
    if (delay?) {
      return DelayBeta?(delay);
    } else if (dist?) {
      return dist!.graftBeta();
    } else {
      return nil;
    }
  }
  
  function graftGamma() -> DelayGamma? {
    if (delay?) {
      return DelayGamma?(delay);
    } else if (dist?) {
      return dist!.graftGamma();
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma() -> DelayInverseGamma? {
    if (delay?) {
      return DelayInverseGamma?(delay);
    } else if (dist?) {
      return dist!.graftInverseGamma();
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if (delay?) {
      m:DelayNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayNormalInverseGamma?(delay))? &&
          (s2 <- σ2.graftInverseGamma())? && m!.σ2 == s2! {
        return m;
      } else {
        return nil;
      }
    } else if (dist?) {
      return dist!.graftNormalInverseGamma(σ2);
    } else {
      return nil;
    }
  }
  
  function graftDirichlet() -> DelayDirichlet? {
    if (delay?) {
      return DelayDirichlet?(delay);
    } else if (dist?) {
      return dist!.graftDirichlet();
    } else {
      return nil;
    }
  }

  function graftRestaurant() -> DelayRestaurant? {
    if (delay?) {
      return DelayRestaurant?(delay);
    } else if (dist?) {
      return dist!.graftRestaurant();
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if (delay?) {
      return DelayMultivariateGaussian?(delay);
    } else if (dist?) {
      return dist!.graftMultivariateGaussian();
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if (delay?) {
      m:DelayMultivariateNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayMultivariateNormalInverseGamma?(delay))? &&
         (s2 <- σ2.graftInverseGamma())? && m!.σ2 == s2! {
        return m;
      } else {
        return nil;
      }
    } else if (dist?) {
      return dist!.graftMultivariateNormalInverseGamma(σ2);
    } else {
      return nil;
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if (delay?) {
      return DelayValue<Integer>?(delay);
    } else if (dist?) {
      return dist!.graftDiscrete();
    } else {
      return nil;
    }
  }
}
