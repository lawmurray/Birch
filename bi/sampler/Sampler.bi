/**
 * Abstract sampler.
 *
 * The Sampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class Sampler {  
  /**
   * Particles.
   */
  x:RandomAccessModel[_];
  
  /**
   * Log-weights.
   */
  w:Real[_];

  /**
   * Ancestor indices.
   */
  a:Integer[_];

  /**
   * Number of samples to draw.
   */
  nsamples:Integer <- 1;
  
  /**
   * Enable verbose reporting on the terminal?
   */
  verbose:Boolean <- true;

  /**
   * Sample the model.
   *
   * Return: a weighted sample.
   */
  function sample() -> (Model, Real);

  /**
   * Set the archetype. This is an instance of the model of interest with
   * zero or more random variates already assigned in order to condition on
   * those assigned values. It represents the target distribution of the
   * inference problem. The sampler will check whether the archetype is of
   * an appropriate type, and may produce an error if this is not the case.
   */
  function setArchetype(m:Model);

  function read(buffer:Buffer) {
    nsamples <-? buffer.get("nsamples", nsamples);
    verbose <-? buffer.get("verbose", verbose);
  }
  
  function write(buffer:Buffer) {
    buffer.set("nsamples", nsamples);
    buffer.set("verbose", verbose);
    configWrite(buffer);
  }
}
