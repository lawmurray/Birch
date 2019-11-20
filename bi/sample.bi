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
    reader:Reader <- Reader(config!);
    reader.read(configBuffer);    
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
  auto inputPath <- input;
  if !inputPath? {
    inputPath <-? configBuffer.getString("input");
  }
  if inputPath? {
    reader:Reader <- Reader(inputPath!);
    inputBuffer:MemoryBuffer;
    reader.read(inputBuffer);
    reader.close();
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
  m <- nil;
  if config? {
    configBuffer.get("sampler", s!);
  }
  assert s?;

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

  /* diagnostic */
  diagnosticWriter:Writer?;
  diagnosticPath:String? <- diagnostic;
  if !diagnosticPath? {
    diagnosticPath <-? configBuffer.getString("diagnostic");
  }
  if diagnosticPath? {
    diagnosticWriter <- Writer(diagnosticPath!);
    diagnosticWriter!.startSequence();
  }

  /* sample */
  m1:Model?;
  w1:Real;
  auto f <- s!.sample(m1!);
  while f? {
    (m1, w1) <- f!;
    if outputWriter? {
      buffer:MemoryBuffer;
      buffer.set(m1!);
      buffer.set("lweight", w1);
      outputWriter!.write(buffer);
      outputWriter!.flush();
    }
    if diagnosticWriter? {
      buffer:MemoryBuffer;
      buffer.set(s!);
      diagnosticWriter!.write(buffer);
      diagnosticWriter!.flush();
    }
  }
  if outputWriter? {
    outputWriter!.endSequence();
    outputWriter!.close();
  }
  if diagnosticWriter? {
    diagnosticWriter!.endSequence();
    diagnosticWriter!.close();
  }
}
