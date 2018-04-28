/*
 *
 */
class TransformMultivariateScaledInverseGamma(A:Boxed<Real[_,_]>,
    σ2:DelayInverseGamma) {
  /**
   * Scale.
   */
  A:Boxed<Real[_,_]> <- A;
  
  /**
   * Inverse gamma.
   */
  σ2:DelayInverseGamma <- σ2;
}
