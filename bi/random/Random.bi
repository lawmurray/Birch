
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
    if (delay?) {
      delay!.realize();
      delay <- nil;
    }
  }

  /**
   * Graft this random variate onto the delayed sampling graph.
   */
  function graft() {
    if (!delay?) {
      delay <- doGraft();
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if (delay?) {
      return delay!.graftGaussian();
    } else {
      return doGraftGaussian();
    }
  }

  function graftAffineGaussianGaussian() -> DelayAffineGaussianGaussian? {
    if (delay?) {
      return delay!.graftAffineGaussianGaussian();
    } else {
      return doGraftAffineGaussianGaussian();
    }
  }

  function graftBeta() -> DelayBeta? {
    if (delay?) {
      return delay!.graftBeta();
    } else {
      return doGraftBeta();
    }
  }

  function graftGamma() -> DelayGamma? {
    if (delay?) {
      return delay!.graftGamma();
    } else {
      return doGraftGamma();
    }
  }

  function graftInverseGamma() -> DelayInverseGamma? {
    if (delay?) {
      return delay!.graftInverseGamma();
    } else {
      return doGraftInverseGamma();
    }
  } 

  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if (delay?) {
      return delay!.graftNormalInverseGamma();
    } else {
      return doGraftNormalInverseGamma();
    }
  }

  function graftDirichlet() -> DelayDirichlet? {
    if (delay?) {
      return delay!.graftDirichlet();
    } else {
      return doGraftDirichlet();
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if (delay?) {
      return delay!.graftMultivariateGaussian();
    } else {
      return doGraftMultivariateGaussian();
    }
  }

  function graftMultivariateAffineGaussianGaussian() ->
      DelayMultivariateAffineGaussianGaussian? {
    if (delay?) {
      return delay!.graftMultivariateAffineGaussianGaussian();
    } else {
      return doGraftMultivariateAffineGaussianGaussian();
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if (delay?) {
      return delay!.graftMultivariateNormalInverseGamma();
    } else {
      return doGraftMultivariateNormalInverseGamma();
    }
  }

  function graftMultivariateNormalInverseGammaGaussian(
      σ2:Expression<Real>) -> DelayMultivariateNormalInverseGammaGaussian? {
    if (delay?) {
      return delay!.graftMultivariateNormalInverseGammaGaussian();
    } else {
      return doGraftMultivariateNormalInverseGammaGaussian();
    }
  }

  function graftMultivariateAffineNormalInverseGammaGaussian(
      σ2:Expression<Real>) ->
      DelayMultivariateAffineNormalInverseGammaGaussian? {
    if (delay?) {
      return delay!.graftMultivariateAffineNormalInverseGammaGaussian();
    } else {
      return doGraftMultivariateAffineNormalInverseGammaGaussian();
    }
  }

  function doGraft() -> Delay? {
    //
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
      DelayMultivariateAffineGaussianGaussian? {
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
