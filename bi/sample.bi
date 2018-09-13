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
 */
program sample(
    model:String,
    method:String <- "ParticleFilter",
    input_file:String?,
    output_file:String?,
    config_file:String?,
    diagnostic_file:String?,
    nsamples:Integer <- 1,
    ncheckpoints:Integer <- 1,
    seed:Integer?,
    verbose:Boolean <- true) {
  /* random number generator */
  if (seed?) {
    global.seed(seed!);
  }

  /* model */
  auto m <- Model?(make(model));
  if (!m?) {
    stderr.print("error: " + model + " must be a subtype of Model with no initialization parameters.\n");
    exit(1);
  }

  /* method */
  auto s <- Method?(make(method));
  if (!s?) {
    stderr.print("error: " + method + " must be a subtype of Method with no initialization parameters.\n");
    exit(1);
  }

  /* I/O */
  input:JSONReader?;
  output:JSONWriter?;
  config:JSONReader?;
  diagnostic:JSONWriter?;
  
  if (input_file?) {
    input <- JSONReader(input_file!);
    input!.read(m!);
  }
  if (output_file?) {
    output <- JSONWriter(output_file!);
  }
  if (config_file?) {
    config <- JSONReader(config_file!);
    config!.read(s!);
  }
  if (diagnostic_file?) {
    diagnostic <- JSONWriter(diagnostic_file!);
  }

  /* sample */
  s!.sample(m!, ncheckpoints, verbose);
  
  /* finalize I/O */
  if (output?) {
    output!.push().write(m!);
    output!.save();
  }
  if (diagnostic?) {
    diagnostic!.push().write(s!);
    diagnostic!.save();
  }
}
