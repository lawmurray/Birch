/**
 * Abstract sampler.
 */
class Sampler {
  /**
   * Run the sampler.
   *
   * - model: Name of the model class.
   * - inputReader: Reader for input.
   * - outputWriter: Writer for output.
   * - diagnosticWriter: Writer for diagnostics.
   * - M: Number of samples.
   * - T: Number of checkpoints.
   * - N: Number of particles.
   * - trigger: Relative ESS below which resampling should be triggered.
   * - verbose: Show debugging messages in the stderr.
   */
  function sample(model:String, inputReader:Reader?, outputWriter:Writer?,
      diagnosticWriter:Writer?, M:Integer, T:Integer, N:Integer,
      trigger:Real, verbose:Boolean);
}
