/**
 * Inference method.
 */
class Method {
  /**
   * Draw a weighted sample from the target distribution
   *
   * - m: Model.
   * - ncheckpoints: Number of checkpoints.
   * - trigger: Relative ESS below which resampling should be triggered.
   *
   * Returns a new sample.
   */
  function sample(m:Model, ncheckpoints:Integer, verbose:Boolean) -> Model {
    //
  }

  /**
   * Read input.
   *
   * - buffer: Buffer.
   *
   * This is typically used to read in method-specific options from a
   * configuration file.
   */
  function read(buffer:Buffer) {
    //
  }
  
  /**
   * Write output.
   *
   * - buffer: Buffer.
   *
   * This is typically used to write method-specific diagnostic information
   * to an output file.
   */
  function write(buffer:Buffer) {
    //
  }
}
