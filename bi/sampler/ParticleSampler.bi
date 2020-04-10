/**
 * Abstract particle sampler.
 *
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
abstract class ParticleSampler {
  /**
   * Latest sample.
   */
  x:Model?;
  
  /**
   * Log weight of latest sample.
   */
  w:Real <- 0.0;

  /**
   * Log normalizing constant estimates for each step.
   */
  lnormalize:Vector<Real>;
  
  /**
   * Effective sample size estimate for each step.
   */
  ess:Vector<Real>;
  
  /**
   * Number of propagations at each step.
   */
  npropagations:Vector<Real>;

  /**
   * Number of samples.
   */
  nsamples:Integer <- 1;

  /**
   * Size. This is the number of calls to `sample(..., Integer)` to be
   * performed after the initial call to `sample(...)`.
   */
  function size() -> Integer {
    return nsamples;
  }
  
  /**
   * Initialize sampler.
   *
   * - filter: Particle filter.
   * - archetype: Archetype. This is an instance of the appropriate model class
   *   that may have one more random variables fixed to known values,
   *   representing the inference problem (or target distribution).
   */
  function sample(filter:ParticleFilter, archetype:Model) {
    //
  }
  
  /**
   * Draw one sample.
   *
   * - filter: Particle filter.
   * - archetype: Archetype. This is an instance of the appropriate model class
   *   that may have one more random variables fixed to known values,
   *   representing the inference problem (or target distribution).
   * - n: The sample number, beginning at 1.
   */
  abstract function sample(filter:ParticleFilter, archetype:Model, n:Integer);
  
  /**
   * Clear diagnostics. These are the record sof ESS, normalizing constants,
   * etc.
   */
  function clearDiagnostics() {
    lnormalize.clear();
    ess.clear();
    npropagations.clear();
  }
  
  /**
   * Push to diagnostics.
   */
  function pushDiagnostics(filter:ParticleFilter) {
    lnormalize.pushBack(filter.lnormalize);
    ess.pushBack(filter.ess);
    npropagations.pushBack(filter.npropagations);
  }
  
  /**
   * Write only the current sample to a buffer.
   */
  function write(buffer:Buffer, n:Integer) {
    buffer.set("sample", clone(x!));
    buffer.set("lweight", w);
    buffer.set("lnormalize", lnormalize);
    buffer.set("ess", ess);
    buffer.set("npropagations", npropagations);
  }
  
  function read(buffer:Buffer) {
    super.read(buffer);
    nsamples <-? buffer.get("nsamples", nsamples);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nsamples", nsamples);
  }
}
