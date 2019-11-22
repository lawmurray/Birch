/**
 * Abstract particle sampler.
 *
 * The Sampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
abstract class ParticleSampler {  
  /**
   * Sample.
   *
   * - model: The model.
   *
   * Yields: a tuple giving, in order:
   *   - sample
   *   - log weight of the sample, and
   *   - for each step of the particle filter run to obtain the sample:
   *     - log normalizing constant estimates,
   *     - effective sample size,
   *     - total number of propagations.
   */
  abstract fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]);
}
