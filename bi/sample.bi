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
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
    diagnostic:String?,
    seed:Integer?) {

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
  model:Model?;
  model <- Model?(configBuffer.get("model", model));
  if !model? {
    error("could not create model; the model class should be given as " + 
        "model.class in the config file, and should derive from Model.");
  }
  auto forwardModel <- Model?(model!);

  /* sampler */
  sampler:Sampler?;
  sampler <- Sampler?(configBuffer.get("sampler", sampler));
  if !sampler? {
    error("could not create sampler; the sampler class should be given as " + 
        "sampler.class in the config file, and should derive from Sampler.");
  }
  
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
    inputBuffer.get(model!);
  }

  /* output */
  outputBuffer:Buffer?;
  outputWriter:Writer?;
  outputPath:String? <- output;
  if !outputPath? {
    outputPath <-? configBuffer.getString("output");
  }
  if outputPath? {
    buffer:MemoryBuffer;
    outputBuffer <- buffer;
    outputWriter <- Writer(outputPath!);
    outputWriter!.startSequence();
  }

  /* diagnostic */
  diagnosticBuffer:Buffer?;
  diagnosticWriter:Writer?;
  diagnosticPath:String? <- diagnostic;
  if !diagnosticPath? {
    diagnosticPath <-? configBuffer.getString("diagnostic");
  }
  if diagnosticPath? {
    buffer:MemoryBuffer;
    diagnosticBuffer <- buffer;
    diagnosticWriter <- Writer(diagnosticPath!);
    diagnosticWriter!.startSequence();
  }

  /* sample */
  auto f <- sampler!.sample(model!);
  while f? {
    if outputWriter? {
      outputWriter!.write(outputBuffer);
      outputWriter!.flush();
    }
    if diagnosticWriter? {
      diagnosticWriter!.write(diagnosticBuffer);
      diagnosticWriter!.flush();
    }
  }
  
  /* finalize output */
  if outputWriter? {
    outputWriter!.endSequence();
    outputWriter!.close();
  }
  if diagnosticWriter? {
    diagnosticWriter!.endSequence();
    diagnosticWriter!.close();
  }
}
