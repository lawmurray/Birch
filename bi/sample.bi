/**
 * Sample from a model.
 *
 * General options.
 *
 *   - `--model`: Name of the model class to use.
 *
 *   - `--method`: Name of the method class to use.
 *
 *   - `--input-file`: Name of the input file, if any.
 *
 *   - `--output-file`: Name of the output file, if any.
 *
 *   - `--diagnostic-file`: Name of the diagnostics file, if any.
 *
 *   - `--nsamples`: Number of samples to draw.
 *
 *   - `--ncheckpoints`: Number of checkpoints for which to run. The
 *     interpretation of this is model-dependent, e.g. for a Markov model
 *     it is the number of states.
 *
 *   - `--seed`: Random number seed. If not provided, random entropy is used.
 *
 *   - `--verbose`: Enable verbose reporting?
 *
 * Particle filter-specific options:
 *
 *   - `--nparticles`: Number of particles to use.
 *
 *   - `--ess-trigger`: Threshold for resampling. Resampling is performed
 *     whenever the effective sample size, as a proportion of `--nparticles`,
 *     drops below this threshold.
 */
program sample(
    model:String <- "Model",
    method:String <- "ParticleFilter",
    input_file:String?,
    output_file:String?,
    diagnostic_file:String?,
    nsamples:Integer <- 1,
    ncheckpoints:Integer <- 1,
    seed:Integer?,
    verbose:Boolean <- true,
    nparticles:Integer <- 1,
    ess_trigger:Real <- 0.7) {
  /* set up I/O */
  input:JSONReader?;
  output:JSONWriter?;
  diagnostic:JSONWriter?;
  
  /* input and output */  
  if (input_file?) {
    input <- JSONReader(input_file!);
    input!.load();
  }
  if (output_file?) {
    output <- JSONWriter(output_file!);
  }
  if (diagnostic_file?) {
    diagnostic <- JSONWriter(diagnostic_file!);
  }
  
  /* seed random number generator */
  if (seed?) {
    global.seed(seed!);
  }
  
  /* set up method */
  sampler:Sampler? <- Sampler?(make(method));
  if (!sampler?) {
    stderr.print("error: " + method + " must be a subtype of Sampler with no initialization parameters.\n");
    exit(1);
  }

  /* sample */
  sampler!.sample(model, input, output, diagnostic, nsamples, ncheckpoints,
      nparticles, ess_trigger, verbose);
  
  /* finalize I/O */
  if (output?) {
    output!.save();
  }
  if (diagnostic?) {
    diagnostic!.save();
  }
}
