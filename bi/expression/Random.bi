
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
    
    dist.associate(this);
    this.dist <- dist;
  }

  /**
   * Get the value of the random variable, forcing realization if necessary.
   */
  function value() -> Value {
    if !x? {
      assert dist?;
      dist!.value();
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
    if (!x?) {
      assert dist?;
      return dist!.graftGaussian();
    } else {
      return nil;
    }
  }
    
  function graftBeta() -> DelayBeta? {
    if (!x?) {
      assert dist?;
      return dist!.graftBeta();
    } else {
      return nil;
    }
  }
  
  function graftGamma() -> DelayGamma? {
    if (!x?) {
      assert dist?;
      return dist!.graftGamma();
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma() -> DelayInverseGamma? {
    if (!x?) {
      assert dist?;
      return dist!.graftInverseGamma();
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if (!x?) {
      assert dist?;
      return dist!.graftNormalInverseGamma(σ2);
    } else {
      return nil;
    }
  }
  
  function graftDirichlet() -> DelayDirichlet? {
    if (!x?) {
      assert dist?;
      return dist!.graftDirichlet();
    } else {
      return nil;
    }
  }

  function graftRestaurant() -> DelayRestaurant? {
    if (!x?) {
      assert dist?;
      return dist!.graftRestaurant();
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if (!x?) {
      assert dist?;
      return dist!.graftMultivariateGaussian();
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if (!x?) {
      assert dist?;
      return dist!.graftMultivariateNormalInverseGamma(σ2);
    } else {
      return nil;
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if (!x?) {
      assert dist?;
      return dist!.graftDiscrete();
    } else {
      return nil;
    }
  }
}
