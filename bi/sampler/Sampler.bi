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
   * Yield: weighted samples.
   */
  abstract fiber sample() -> (Model, Real);

  /**
   * Set the archetype. This is an instance of the model of interest with
   * zero or more random variates already assigned in order to condition on
   * those assigned values. It represents the target distribution of the
   * inference problem. The sampler will check whether the archetype is of
   * an appropriate type, and may produce an error if this is not the case.
   */
  abstract function setArchetype(m:Model);

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
