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

  function graftGaussian() -> DelayGaussian? {
    prune();
    return doGraftGaussian();
  }

  function graftAffineGaussianGaussian() -> DelayAffineGaussianGaussian? {
    prune();
    return doGraftAffineGaussianGaussian();
  }

  function graftBeta() -> DelayBeta? {
    prune();
    return doGraftBeta();
  }

  function graftGamma() -> DelayGamma? {
    prune();
    return doGraftGamma();
  }

  function graftInverseGamma() -> DelayInverseGamma? {
    prune();
    return doGraftInverseGamma();
  } 

  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    prune();
    return doGraftNormalInverseGamma();
  }

  function graftDirichlet() -> DelayDirichlet? {
    prune();
    return doGraftDirichlet();
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    prune();
    return doGraftMultivariateGaussian();
  }

  function graftMultivariateAffineGaussianGaussian() ->
      DelayAffineMultivariateGaussianGaussian? {
    prune();
    return doGraftMultivariateAffineGaussianGaussian();
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    prune();
    return doGraftMultivariateNormalInverseGamma();
  }

  function graftMultivariateNormalInverseGammaGaussian(
      σ2:Expression<Real>) -> DelayMultivariateNormalInverseGammaGaussian? {
    prune();
    return doGraftMultivariateNormalInverseGammaGaussian();
  }

  function graftMultivariateAffineNormalInverseGammaGaussian(
      σ2:Expression<Real>) ->
      DelayMultivariateAffineNormalInverseGammaGaussian? {
    prune();
    return doGraftMultivariateAffineNormalInverseGammaGaussian();
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
