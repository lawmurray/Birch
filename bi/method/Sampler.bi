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
    auto nsamples1 <- buffer.get("nsamples", nsamples);
    if nsamples1? {
      nsamples <- nsamples1!;
    }
    ncheckpoints <- buffer.getInteger("ncheckpoints");
    auto verbose1 <- buffer.get("verbose", verbose);
    if verbose1? {
      verbose <- verbose1!;
    }
  }
  
  function write(buffer:Buffer) {
    buffer.set("nsamples", nsamples);
    buffer.set("ncheckpoints", ncheckpoints);
    buffer.set("verbose", verbose);
  }
}
