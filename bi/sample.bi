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
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
    diagnostic:String?,
    seed:Integer?) {
  /* random number generator */
  if (seed?) {
    global.seed(seed!);
  }

  /* input */
  inputBuffer:JSONBuffer;
  configBuffer:JSONBuffer;
  if (input?) {
    inputBuffer.load(input!);
  }
  if (config?) {
    configBuffer.load(config!);
  }
    
  /* sampler */
  sampler:Sampler?;
  className:String?;
  buffer:Buffer? <- configBuffer.getObject("sampler");
  if (buffer?) {
    className <- buffer!.getString("class");
  }
  if (!className?) {
    className <- "ParticleFilter";
  }
  sampler <- Sampler?(make(className!));
  if (!sampler?) {
    error(className! + " is not a subtype of Sampler.");
  }
  configBuffer.get("sampler", sampler!);

  /* model */
  model:Model?;
  className <- nil;
  buffer <- configBuffer.getObject("model");
  if (buffer?) {
    className <- buffer!.getString("class");
  }
  if (className?) {
    model <- Model?(make(className!));
    if (!model?) {
      error(className! + " is not a subtype of Model.");
    }
  } else {
    error("no model class specified, this should be given as model.class in the config file.");
  }
  //configBuffer.get("model", model!);
  inputBuffer.get(model!);
  
  /* output */
  outputBuffer:JSONBuffer;
  diagnosticBuffer:JSONBuffer;
  if (sampler!.nsamples > 1) {
    outputBuffer.setArray();
    diagnosticBuffer.setArray();
  }
  
  /* sample */
  for n:Integer in 1..sampler!.nsamples {  
    m1:Model?;
    w1:Real;
    (m1, w1) <- sampler!.sample(model!);
        
    if (sampler!.nsamples > 1) {
      auto buffer <- outputBuffer.push();
      buffer.set(m1!);
      buffer.set("lweight", w1);
      buffer <- diagnosticBuffer.push();
      buffer.set(sampler!);
    } else {
      outputBuffer.set(m1!);
      outputBuffer.set("lweight", w1);
      diagnosticBuffer.set(sampler!);
    }
  }
  if (output?) {
    outputBuffer.save(output!);
  }
  if (diagnostic?) {
    diagnosticBuffer.save(diagnostic!);
  }
}
