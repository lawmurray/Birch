/**
 * Sample from a model.
 *
 * General options.
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
    m!.read(input);
  }
  if (output_file?) {
    output <- JSONWriter(output_file!);
    if (nsamples > 1) {
      output!.setArray();
    }
  }
  if (config_file?) {
    config <- JSONReader(config_file!);
    s!.read(config);
  }
  if (diagnostic_file?) {
    diagnostic <- JSONWriter(diagnostic_file!);
    if (nsamples > 1) {
      diagnostic!.setArray();
    }
  }

  /* sample */
  for i:Integer in 1..nsamples {
    m <- s!.sample(m!, ncheckpoints, verbose);
      
    if (output?) {
      if (nsamples > 1) {
        m!.write(output!.push());
      } else {
        m!.write(output!);
      }
    }
    if (diagnostic?) {
      if (nsamples > 1) {
        s!.write(diagnostic!.push());
      } else {
        s!.write(diagnostic!);
      }
    }
  }
  if (output?) {
    output!.save();
  }
  if (diagnostic?) {
    diagnostic!.save();
  }
}
