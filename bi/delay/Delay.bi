/**
 * Interface for delayed sampling of random variables.
 */
class Delay {
  /**
   * Parent.
   */
  parent:Delay?;
  
  /**
   * Child, if one exists and it is on the $M$-path.
   */
  child:Delay&;
  
  /**
   * Realize (simulate or observe).
   */
  function realize();
  
  /**
   * Prune the $M$-path from below this node.
   */
  function prune() {
    child:Delay? <- this.child;
    if (child?) {
      child!.prune();
      child!.realize();
      child <- nil;
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
