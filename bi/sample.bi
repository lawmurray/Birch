/**
 * Sample from a model.
 *
 * - `--input`: Name of the input file, if any. Alternatively (preferably),
 *   provide this as `input` in the configuration file.
 *
 * - `--output`: Name of the output file, if any. Alternatively (preferably),
 *   provide this as `output` in the configuration file.
 *
 * - `--config`: Name of the configuration file, if any. Alternatively
 *   (preferably), provide this as `config` in the configuration file.
 *
 * - `--diagnostic`: Name of the diagnostic file, if any. Alternatively
 *   (preferably), provide this as a `diagnostic` in the configuration file.
 *
 * - `--seed`: Random number seed. Alternatively (preferably), provide this as
 *   `seed` in the configuration file. If not provided, random entropy is used.
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

  /* config */
  configBuffer:MemoryBuffer;
  if config? {
    configBuffer.load(config!);
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

  /* input */
  inputFile:String?;
  inputBuffer:MemoryBuffer;
  if input? {
    inputFile <- input!;
  } else if config? {
    auto buffer <- configBuffer.getString("input");
    if buffer? {
      inputFile <- buffer!;
    }
  }
  if inputFile? {
    inputBuffer.load(inputFile!);
    inputBuffer.get(m!);
  }

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
  s!.setArchetype(m!);
  m <- nil;
  if config? {
    configBuffer.get("sampler", s!);
  }
  assert s?;

  /* output */
  outputBuffer:MemoryBuffer;
  outputBuffer.setArray();
  outputFile:String?;
  if output? {
    outputFile <- output!;
  } else if config? {
    auto buffer <- configBuffer.getString("output");
    if buffer? {
      outputFile <- buffer!;
    }
  }

  /* diagnostic */
  diagnosticBuffer:MemoryBuffer;
  diagnosticBuffer.setArray();
  diagnosticFile:String?;
  if diagnostic? {
    diagnosticFile <- diagnostic!;
  } else if config? {
    auto buffer <- configBuffer.getString("diagnostic");
    if buffer? {
      diagnosticFile <- buffer!;
    }
  }


  /* sample */
  m1:Model?;
  w1:Real;
  auto f <- s!.sample();
  while f? {
    (m1, w1) <- f!;
    if outputFile? {
      auto buffer <- outputBuffer.push();
      buffer.set(m1!);
      buffer.set("lweight", w1);
    }
    if diagnosticFile? {
      auto buffer <- diagnosticBuffer.push();
      buffer.set(s!);
    }
  }
  if outputFile? {
    outputBuffer.save(outputFile!);
  }
  if diagnosticFile? {
    diagnosticBuffer.save(diagnosticFile!);
  }
}
