/**
 * Inference method.
 */
class Method {
  /**
   * Draw a weighted sample from the target distribution
   *
   * - v: Variate.
   * - m: Model.
   * - ncheckpoints: Number of checkpoints.
   * - trigger: Relative ESS below which resampling should be triggered.
   *
   * Returns a new sample.
   */
  function sample(v:Variate, m:Model, ncheckpoints:Integer, verbose:Boolean) -> Variate {
    //
  }

  /**
   * Read input.
   *
   * - reader: Reader.
   *
   * This is typically used to read in method-specific options from a
   * configuration file.
   */
  function read(reader:Reader) {
    //
  }
  
  /**
   * Write output.
   *
   * - writer: Writer.
   *
   * This is typically used to write method-specific diagnostic information
   * to an output file.
   */
  function write(writer:Writer) {
    //
  }
}
