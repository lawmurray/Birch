/**
 * Sample from a model.
 *
 * General options.
 *
 *   - `--model`: Name of the model class to use.
 *
 *   - `--input-file`: Name of the input file, if any.
 *
 *   - `--output-file`: Name of the output file, if any.
 *
 *   - `--nsamples`: Number of samples to draw.
 *
 *   - `--ncheckpoints`: Number of checkpoints for which to run. The
 *     interpretation of this is model-dependent, e.g. for a Markov model
 *     it is the number of states.
 *
 * SMC method-specific options:
 *
 *   - `--nparticles`: Number of particles to use.
 *
 *   - `--ess-trigger`: Threshold for resampling. Resampling is performed
 *     whenever the effective sample size, as a proportion of `--nparticles`,
 *     drops below this threshold.
 */
program sample(
    model:String <- "Model",
    input_file:String?,
    output_file:String?,
    nsamples:Integer <- 1,
    ncheckpoints:Integer <- 1,
    nparticles:Integer <- 1,
    ess_trigger:Real <- 0.7) {
  input:JSONReader;
  if (input_file?) {
    input.load(input_file!);
  }
  output:JSONWriter;  
  method:SMC;
  for (n:Integer in 1..nsamples) {
    method.simulate(model, input, output, ncheckpoints, nparticles,
        ess_trigger);
  }
  if (output_file?) {
    output.save(output_file!);
  }
}
