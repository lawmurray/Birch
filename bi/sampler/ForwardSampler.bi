/**
 * Abstract sampler for a ForwardModel.
 *
 * The Sampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
abstract class ForwardSampler {  
  /**
   * Sample the model.
   *
   * Yield: weighted samples.
   */
  abstract fiber sample(model:ForwardModel) -> (ForwardModel, Real);
}
