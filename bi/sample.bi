/**
 * Sample from a model.
 *
 * General options.
 *
 *   - `--variate`: Name of the variate class.
 *
 *   - `--model`: Name of the model class.
 *
 *   - `--method`: Name of the method class.
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
 */
program sample(
    variate:String,
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

  /* variate */
  auto v <- Variate?(make(variate));
  if (!v?) {
    stderr.print("error: " + variate + " must be a subtype of Variate with no initialization parameters.\n");
    exit(1);
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
    v!.read(input);
  }
  if (output_file?) {
    output <- JSONWriter(output_file!);
    output!.setArray();
  }
  if (config_file?) {
    config <- JSONReader(config_file!);
    s!.read(config);
  }
  if (diagnostic_file?) {
    diagnostic <- JSONWriter(diagnostic_file!);
    diagnostic!.setArray();
  }

  /* sample */
  for i:Integer in 1..nsamples {
    v <- s!.sample(v!, m!, ncheckpoints, verbose);
      
    if (output?) {
      v!.write(output!.push());
      output!.save();
    }
    if (diagnostic?) {
      s!.write(diagnostic!.push());
      diagnostic!.save();
    }
  }
}
