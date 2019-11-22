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
   * Number of samples.
   */
  nsamples:Integer <- 1;

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
  
  function read(buffer:Buffer) {
    nsamples <-? buffer.get("nsamples", nsamples);
  }
  
  function write(buffer:Buffer) {
    buffer.set("nsamples", nsamples);
  }
}
