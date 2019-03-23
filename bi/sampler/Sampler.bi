/**
 * Abstract sampler.
 */
class Sampler {  
  /**
   * Number of samples to draw.
   */
  nsamples:Integer <- 1;

  /**
   * Number of steps for which to run the model. The interpretation of
   * this is model-dependent, e.g. for MarkovModel or HiddenSpaceModel it is
   * the number of states, in other cases it may be the number of
   * observations. If not given, the model to run to termination.
   */
  nsteps:Integer?;
  
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
    auto nsamples1 <- buffer.get("nsamples", nsamples);
    if nsamples1? {
      nsamples <- nsamples1!;
    }
    nsteps <- buffer.getInteger("nsteps");
    auto verbose1 <- buffer.get("verbose", verbose);
    if verbose1? {
      verbose <- verbose1!;
    }
  }
  
  function write(buffer:Buffer) {
    buffer.set("nsamples", nsamples);
    buffer.set("nsteps", nsteps);
    buffer.set("verbose", verbose);
    configWrite(buffer);
  }
}
