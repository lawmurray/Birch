/**
 * Lazily-evaluated expression.
 *
 * - Value: Value type.
 */
class Expression<Value> {  
  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }

  /**
   * Value evaluation.
   */
  function value() -> Value {
    assert false;
  }
  
  /**
   * Boxed value evaluation.
   */
  function boxed() -> Boxed<Value> {
    return Boxed(value());
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGaussian() -> DelayGaussian? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftAffineGaussian() -> TransformAffineGaussian? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftAffineGaussianGaussian() -> DelayAffineGaussianGaussian? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftBeta() -> DelayBeta? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGamma() -> DelayGamma? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftInverseGamma() -> DelayInverseGamma? {
    return nil;
  } 

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftScaledInverseGamma(σ2:Expression<Real>) -> 
      TransformScaledInverseGamma? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftAffineNormalInverseGamma(σ2:Expression<Real>) ->
      TransformAffineNormalInverseGamma? {
    return nil;
  }
  
  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDirichlet() -> DelayDirichlet? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateAffineGaussian() ->
      TransformMultivariateAffineGaussian? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateAffineGaussianGaussian() ->
      DelayMultivariateAffineGaussianGaussian? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateScaledInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateScaledInverseGamma? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateNormalInverseGammaGaussian(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGammaGaussian? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateAffineNormalInverseGamma? {
    return nil;
  }

  /**
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: Inverse gamma expression that must appear.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateAffineNormalInverseGammaGaussian(σ2:Expression<Real>) ->
      DelayMultivariateAffineNormalInverseGammaGaussian? {
    return nil;
  }
}
