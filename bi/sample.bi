/**
 * Sample from a model.
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
 *
 * - `--model`: Name of the model class. Alternatively (preferably), provide
 *   this as `model.class` in the configuration file.
 *
 * - `--sampler`: Name of the sampler class. Alternatively (preferably),
 *   provide this as `sampler.class` in the configuration file. If the
 *   sampler is not provided through either of these mechanisms, a default
 *   is chosen according to the model class.
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
    diagnostic:String?,
    seed:Integer?,
    model:String?,
    sampler:String?) {
  /* random number generator */
  if seed? {
    global.seed(seed!);
  }
  
  /* config */
  configBuffer:JSONBuffer;
  if config? {
    configBuffer.load(config!);
  }
    
  /* model */
  m:Model?;
  modelClass:String?;
  if model? {
    modelClass <- model!;
  } else if config? {
    auto buffer <- configBuffer.getObject("model");
    if buffer? {
      modelClass <- buffer!.getString("class");
    }
  }
  if modelClass? {
    m <- Model?(make(modelClass!));
    if !m? {
      error(modelClass! + " is not a subtype of Model.");
    }
    if config? {
      configBuffer.get("model", m!);
    }
  } else {
    error("no model class specified, this should be given using the --model option, or as model.class in the config file.");
  }
  assert m?;

  /* sampler */
  s:Sampler?;
  samplerClass:String?;
  if sampler? {
    samplerClass <- sampler!;
  } else if config? {
    auto buffer <- configBuffer.getObject("sampler");
    if buffer? {
      samplerClass <- buffer!.getString("class");
    }
  }
  if !samplerClass? {
    /* determine an appropriate default */
    auto forwardModel <- ForwardModel?(m!);
    if forwardModel? {
      samplerClass <- "ParticleFilter";
    } else {
      samplerClass <- "ImportanceSampler";
    }
  }
  s <- Sampler?(make(samplerClass!));
  if !s? {
    error(samplerClass! + " is not a subtype of Sampler.");
  }
  if config? {
    configBuffer.get("sampler", s!);
  }
  assert s?;
  s!.setArchetype(m!);
  
  /* input */
  inputBuffer:JSONBuffer;
  if input? {
    inputBuffer.load(input!);
    inputBuffer.get(m!);
  }
  
  /* output */
  outputBuffer:JSONBuffer;
  diagnosticBuffer:JSONBuffer;
  if s!.nsamples > 1 {
    outputBuffer.setArray();
    diagnosticBuffer.setArray();
  }
  
  /* sample */
  for n:Integer in 1..s!.nsamples {  
    m1:Model?;
    w1:Real;
    (m1, w1) <- s!.sample();
        
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
