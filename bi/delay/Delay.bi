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
    return doGraftNormalInverseGamma(σ2);
  }

  function graftDirichlet() -> DelayDirichlet? {
    prune();
    return doGraftDirichlet();
  }

  function graftRestaurant() -> DelayRestaurant? {
    prune();
    return doGraftRestaurant();
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    prune();
    return doGraftMultivariateGaussian();
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    prune();
    return doGraftMultivariateNormalInverseGamma(σ2);
  }

  function doGraftGaussian() -> DelayGaussian? {
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

  function doGraftRestaurant() -> DelayRestaurant? {
    return nil;
  }

  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return nil;
  }

  function doGraftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    return nil;
  }
}
