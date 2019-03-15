/**
 * Abstract sampler.
 */
class Sampler {  
  /**
   * Number of samples to draw.
   */
  nsamples:Integer <- 1;

  /**
   * Number of checkpoints for which to run the model. The interpretation of
   * this is model-dependent, e.g. for MarkovModel or StateSpaceModel it is
   * the number of states, in other cases it may be the number of
   * observations. If not given, the model to run to termination.
   */
  ncheckpoints:Integer?;
  
  /**
   * Enable verbose reporting on the terminal?
   */
  verbose:Boolean <- true;

  /**
   * Sample the model.
   *
   * - m: Archetype.
   *
   * Return: a weighted sample.
   */
  function sample(m:Model) -> (Model, Real);

  function read(buffer:Buffer) {
    nsamples <-? buffer.get("nsamples", nsamples);
    ncheckpoints <-? buffer.get("ncheckpoints", ncheckpoints);
    verbose <-? buffer.get("verbose", verbose);
  }
  
  function write(buffer:Buffer) {
    buffer.set("nsamples", nsamples);
    buffer.set("ncheckpoints", ncheckpoints);
    buffer.set("verbose", verbose);
    configWrite(buffer);
  }
}
