/**
 * Sample from a model.
 *
 *
 * - `--input`: Name of the input file, if any.
 *
 * - `--output`: Name of the output file, if any.
 *
 * - `--config`: Name of the configuration file, if any.
 *
 * - `--diagnostic`: Name of the diagnostic file, if any.
 *
 * - `--model`: Name of the model class. Alternatively, provide this as
 *   `model.class` in the configuration file.
 *
 * - `--sampler`: Name of the sampler class. Alternatively, provide this as
 *   `sampler.class` in the configuration file.
 *
 * - `--seed`: Random number seed. If not provided, random entropy is used.
 */
program sample(
    model:String?,
    sampler:String?,
    input:String?,
    output:String?,
    config:String?,
    diagnostic:String?,
    seed:Integer?) {
  /* random number generator */
  if seed? {
    global.seed(seed!);
  }
  
  /* config */
  configBuffer:MemoryBuffer;
  if config? {
    configBuffer.load(config!);
  }
    
  /* sampler */
  s:Sampler?;
  className:String?;
  if sampler? {
    className <- sampler!;
  } else if config? {
    auto buffer <- configBuffer.getObject("sampler");
    if buffer? {
      className <- buffer!.getString("class");
    }
  }
  if !className? {
    className <- "ParticleFilter";
  }
  s <- Sampler?(make(className!));
  if !s? {
    error(className! + " is not a subtype of Sampler.");
  }
  if config? {
    configBuffer.get("sampler", s!);
  }
  
  /* model */
  m:Model?;
  className <- nil;
  if model? {
    className <- model!;
  } else if config? {
    auto buffer <- configBuffer.getObject("model");
    if buffer? {
      className <- buffer!.getString("class");
    }
  }
  if className? {
    m <- Model?(make(className!));
    if !m? {
      error(className! + " is not a subtype of Model.");
    }
    if config? {
      configBuffer.get("model", m!);
    }
  } else {
    error("no model class specified, this should be given using the --model option, or as model.class in the config file.");
  }
  
  /* input */
  inputBuffer:MemoryBuffer;
  if input? {
    inputBuffer.load(input!);
    inputBuffer.get(m!);
  }
  
  /* output */
  outputBuffer:MemoryBuffer;
  diagnosticBuffer:MemoryBuffer;
  if s!.nsamples > 1 {
    outputBuffer.setArray();
    diagnosticBuffer.setArray();
  }
  
  /* sample */
  for n:Integer in 1..s!.nsamples {  
    m1:Model?;
    w1:Real;
    (m1, w1) <- s!.sample(m!);
        
    if s!.nsamples > 1 {
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
  if output? {
    outputBuffer.save(output!);
  }
  if diagnostic? {
    diagnosticBuffer.save(diagnostic!);
  }
}
