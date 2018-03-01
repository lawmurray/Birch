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
    nparticles:Integer <- 1,
    ess_trigger:Real <- 0.7) {
  /* set up I/O */
  input:JSONReader?;
  output:JSONWriter?;
  diagnostic:JSONWriter?;
  
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
  
  /* set up method */
  filter:ParticleFilter? <- ParticleFilter?(make(method));
  if (!filter?) {
    stderr.print("error: " + method + " must be a subtype of ParticleFilter with no initialization parameters.\n");
    exit(1);
  }

  /* sample */
  for (n:Integer in 1..nsamples) {
    filter!.filter(model, input, output, diagnostic, ncheckpoints,
        nparticles, ess_trigger);
  }
  
  /* finalize I/O */
  if (output?) {
    output!.save();
  }
  if (diagnostic?) {
    diagnostic!.save();
  }
}
