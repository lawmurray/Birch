
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
   * Weight.
   */
  w:Real?;

  /**
   * Associated node in delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    this.x <- x;
    realize();
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    if (x?) {
      this.x <- x!;
      realize();
    }
  }

  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    return !x?;
  }

  /**
   * Get the value of the random variable, forcing realization if necessary.
   */
  function value() -> Value {
    realize();
    assert x?;
    return x!;
  }

  /**
   * Simulate the random variable.
   */
  function simulate() -> Value {
    realize();
    assert x?;
    return x!;
  }

  /**
   * Observe the random variable.
   *
   * - x: The observed value.
   *
   * Return: the log likelihood.
   */
  function observe(x:Value) -> Real {
    this.x <- x;
    realize();
    assert w?;
    return w!;
  }

  /**
   * Realize a value for this random variable.
   */
  function realize() {
    graft();
    delay!.realize();
    delay <- nil;
  }

  /**
   * Graft this random variate onto the delayed sampling graph.
   */
  function graft() {
    if (!delay?) {
      doGraft();
      assert delay?;
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if (delay?) {
      return delay!.doGraftGaussian();
    } else {
      return doGraftGaussian();
    }
  }

  function graftAffineGaussianGaussian() -> DelayAffineGaussianGaussian? {
    if (delay?) {
      return delay!.doGraftAffineGaussianGaussian();
    } else {
      return doGraftAffineGaussianGaussian();
    }
  }

  function graftBeta() -> DelayBeta? {
    if (delay?) {
      return delay!.doGraftBeta();
    } else {
      return doGraftBeta();
    }
  }

  function graftGamma() -> DelayGamma? {
    if (delay?) {
      return delay!.doGraftGamma();
    } else {
      return doGraftGamma();
    }
  }

  function graftInverseGamma() -> DelayInverseGamma? {
    if (delay?) {
      return delay!.doGraftInverseGamma();
    } else {
      return doGraftInverseGamma();
    }
  } 

  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if (delay?) {
      return delay!.doGraftNormalInverseGamma();
    } else {
      return doGraftNormalInverseGamma();
    }
  }

  function graftDirichlet() -> DelayDirichlet? {
    if (delay?) {
      return delay!.doGraftDirichlet();
    } else {
      return doGraftDirichlet();
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if (delay?) {
      return delay!.doGraftMultivariateGaussian();
    } else {
      return doGraftMultivariateGaussian();
    }
  }

  function graftMultivariateAffineGaussianGaussian() ->
      DelayAffineMultivariateGaussianGaussian? {
    if (delay?) {
      return delay!.doGraftMultivariateAffineGaussianGaussian();
    } else {
      return doGraftMultivariateAffineGaussianGaussian();
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if (delay?) {
      return delay!.doGraftMultivariateNormalInverseGamma();
    } else {
      return doGraftMultivariateNormalInverseGamma();
    }
  }

  function graftMultivariateNormalInverseGammaGaussian(
      σ2:Expression<Real>) -> DelayMultivariateNormalInverseGammaGaussian? {
    if (delay?) {
      return delay!.doGraftMultivariateNormalInverseGammaGaussian();
    } else {
      return doGraftMultivariateNormalInverseGammaGaussian();
    }
  }

  function graftMultivariateAffineNormalInverseGammaGaussian(
      σ2:Expression<Real>) ->
      DelayMultivariateAffineNormalInverseGammaGaussian? {
    if (delay?) {
      return delay!.doGraftMultivariateAffineNormalInverseGammaGaussian();
    } else {
      return doGraftMultivariateAffineNormalInverseGammaGaussian();
    }
  }

  function doGraft() {
    assert false;
  }

  function doGraftGaussian() -> DelayGaussian? {
    return nil;
  }

  function doGraftAffineGaussianGaussian() -> DelayAffineGaussianGaussian? {
    return nil;
  }

  function doGraftBeta() -> DelayBeta? {
    return nil;
  }

  function doGraftGamma() -> DelayGamma? {
    return nil;
  }

  function doGraftInverseGamma() -> DelayInverseGamma? {
    return nil;
  } 

  function doGraftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    return nil;
  }

  function doGraftDirichlet() -> DelayDirichlet? {
    return nil;
  }

  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return nil;
  }

  function doGraftMultivariateAffineGaussianGaussian() ->
      DelayAffineMultivariateGaussianGaussian? {
    return nil;
  }

  function doGraftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    return nil;
  }

  function doGraftMultivariateNormalInverseGammaGaussian(
      σ2:Expression<Real>) -> DelayMultivariateNormalInverseGammaGaussian? {
    return nil;
  }

  function doGraftMultivariateAffineNormalInverseGammaGaussian(
      σ2:Expression<Real>) ->
      DelayMultivariateAffineNormalInverseGammaGaussian? {
    return nil;
  }
}
