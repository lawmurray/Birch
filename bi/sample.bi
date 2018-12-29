/**
 * Sample from a model.
 *
 * - `--model`: Name of the model class. Should be a subtype of Model.
 *
 * - `--sampler`: Name of the sampler class. Should be a subtype of Sampler.
 *
 * - `--input`: Name of the input file, if any.
 *
 * - `--output`: Name of the output file, if any.
 *
 * - `--config`: Name of the configuration file, if any.
 *
 * - `--diagnostic`: Name of the diagnostic file, if any.
 *
 * - `--seed`: Random number seed. If not provided, random entropy is used.
 */
program sample(
    model:String,
    sampler:String <- "ParticleFilter",
    input:String?,
    output:String?,
    config:String?,
    diagnostic:String?,
    seed:Integer?) {
  /* seed random number generator */
  if (seed?) {
    global.seed(seed!);
  }
    
  /* model */
  auto m <- Model?(make(model));
  if (!m?) {
    error(model + " must be a subtype of Model with no initialization parameters.");
  }

  /* method */
  auto s <- Sampler?(make(sampler));
  if (!s?) {
    error(sampler + " must be a subtype of Sampler with no initialization parameters.");
  }

  /* input */
  inputBuffer:JSONBuffer;
  if (input?) {
    inputBuffer.load(input!);
    inputBuffer.get(m!);
  }

  /* config */
  configBuffer:JSONBuffer;
  if (config?) {
    configBuffer.load(config!);
    configBuffer.get(s!);
  }
  
  /* output */
  outputBuffer:JSONBuffer;
  if (s!.nsamples > 1) {
    outputBuffer.setArray();
  }
  
  /* diagnostics */
  diagnosticBuffer:JSONBuffer;
  if (s!.nsamples > 1) {
    diagnosticBuffer.setArray();
  }
  
  /* sample */
  auto f <- s!.sample(m!);
  while (f?) {
    m1:Model?;
    w1:Real;
    (m1, w1) <- f!;
    
    if (s!.nsamples > 1) {
      auto buffer <- outputBuffer.push();
      buffer.set(m1!);
      buffer.set("lweight", w1);
      buffer <- diagnosticBuffer.push();
      buffer.set(s!);
    } else {
      outputBuffer.set(m1!);
      outputBuffer.set("lweight", w1);
      diagnosticBuffer.set(s!);
    }
  }
  if (output?) {
    outputBuffer.save(output!);
  }
  if (diagnostic?) {
    diagnosticBuffer.save(diagnostic!);
  }
}
