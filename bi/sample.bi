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
  /* set up input and output */
  input:JSONReader?;
  output:JSONWriter?;
  if (input_file?) {
    input <- JSONReader(input_file!);
    input!.load();
  }
  if (output_file?) {
    output <- JSONWriter(output_file!);
    output!.setArray();
  }
  
  /* sample */
  method:SMC;
  x:Model;
  Z:Real;
  for (n:Integer in 1..nsamples) {
    method.simulate(model, input, output, ncheckpoints, nparticles, ess_trigger);
  }
  
  /* finalize output */
  if (output?) {
    output!.save();
  }
}
