/**
 * Sample a model.
 *
 * - `--config`: Name of the configuration file, if any.
 *
 * - `--input`: Name of the input file, if any. Alternatively, provide this
 *   as `input` in the configuration file.
 *
 * - `--output`: Name of the output file, if any. Alternatively, provide this
 *   as `output` in the configuration file.
 *
 * - `--model`: Name of the model class, if any. Alternatively, provide this
 *   as `model.class` in the configuration file.
 *
 * - `--seed`: Random number seed. Alternatively, provide this as `seed` in
 *   the configuration file. If not provided, random entropy is used.
 *
 * - `--quiet`: Don't display a progress bar.
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
    model:String?,
    seed:Integer?,
    quiet:Boolean <- false) {
  /* config */
  configBuffer:MemoryBuffer;
  if config? {
    auto reader <- Reader(config!);
    configBuffer <- reader.scan();    
    reader.close();
  }

  /* random number generator */
  if seed? {
    global.seed(seed!);
  } else if config? {
    auto buffer <- configBuffer.getInteger("seed");
    if buffer? {
      global.seed(buffer!);
    }
  } else {
    global.seed();
  }

  /* model */
  auto buffer <- configBuffer.getObject("model");
  if !buffer? {
    buffer <- configBuffer.setObject("model");
  }
  if !buffer!.getString("class")? && model? {
    buffer!.setString("class", model!);
  }
  auto archetype <- Model?(make(buffer));
  if !archetype? {
    error("could not create model; the model class should be given as " + 
        "model.class in the config file, and should derive from Model.");
  }

  /* sampler */
  buffer <- configBuffer.getObject("sampler");
  if !buffer? {
    buffer <- configBuffer.setObject("sampler");
  }
  if !buffer!.getString("class")? {
    buffer!.setString("class", "ParticleMarginalImportanceSampler");
  }
  auto sampler <- ParticleSampler?(make(buffer));
  if !sampler? {
    error("could not create sampler; the sampler class should be given as " + 
        "sampler.class in the config file, and should derive from ParticleSampler.");
  }

  /* filter */
  buffer <- configBuffer.getObject("filter");
  if !buffer? {
    buffer <- configBuffer.setObject("filter");
  }
  if !buffer!.getString("class")? {
    if ConditionalParticleSampler?(sampler)? {
      buffer!.setString("class", "ConditionalParticleFilter");
    } else {
      buffer!.setString("class", "ParticleFilter");
    }
  }
  auto filter <- ParticleFilter?(make(buffer));
  if !filter? {
    error("could not create filter; the filter class should be given as " + 
        "filter.class in the config file, and should derive from ParticleFilter.");
  }
  
  /* input */
  auto inputPath <- input;
  if !inputPath? {
    inputPath <-? configBuffer.getString("input");
  }
  if inputPath? {
    auto reader <- Reader(inputPath!);
    auto inputBuffer <- reader.scan();
    reader.close();
    inputBuffer.get(archetype!);
  }

  /* output */
  outputWriter:Writer?;
  outputPath:String? <- output;
  if !outputPath? {
    outputPath <-? configBuffer.getString("output");
  }
  if outputPath? {
    outputWriter <- Writer(outputPath!);
    outputWriter!.startSequence();
  }

  /* progress bar */
  bar:ProgressBar;
  if !quiet {
    bar.update(0.0);
  }

  /* sample */
  sampler!.sample(filter!, archetype!);
  for n in 1..sampler!.size() {
    sampler!.sample(filter!, archetype!, n);
    
    if outputWriter? {
      buffer:MemoryBuffer;
      sampler!.write(buffer, n);      
      outputWriter!.print(buffer);
      outputWriter!.flush();
    }
    if !quiet {
      bar.update(Real(n)/sampler!.nsamples);
    }
  }
  
  /* finalize output */
  if outputWriter? {
    outputWriter!.endSequence();
    outputWriter!.close();
  }
}
