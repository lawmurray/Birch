/**
 * Abstract sampler.
 *
 * The Sampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
abstract class Sampler {  
  /**
   * Sample the model.
   *
   * - model: The model.
   *
   * Yield: weighted samples.
   */
  abstract fiber sample(model:Model) -> (Model, Real);
}
