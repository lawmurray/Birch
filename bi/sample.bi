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
 * - `--seed`: Random number seed. Alternatively, provide this as `seed` in
 *   the configuration file. If not provided, random entropy is used.
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
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

  /* sampler */
  sampler:ParticleSampler?;
  sampler <- ParticleSampler?(configBuffer.get("sampler", sampler));
  if !sampler? {
    error("could not create sampler; the sampler class should be given as " + 
        "sampler.class in the config file, and should derive from ParticleSampler.");
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
  outputWriter:Writer?;
  outputPath:String? <- output;
  if !outputPath? {
    outputPath <-? configBuffer.getString("output");
  }
  if outputPath? {
    outputWriter <- Writer(outputPath!);
    outputWriter!.startSequence();
  }

  /* sample */
  auto f <- sampler!.sample(model!);
  while f? {
    if outputWriter? {
      sample:Model;
      lweight:Real;
      lnormalizer:Real[_];
      ess:Real[_];
      npropagations:Integer[_];
      (sample, lweight, lnormalizer, ess, npropagations) <- f!;
    
      buffer:MemoryBuffer;
	  buffer.set("sample", sample);
	  buffer.set("lweight", lweight);
	  buffer.set("lnormalizer", lnormalizer);
	  buffer.set("ess", ess);
	  buffer.set("npropagations", npropagations);
      outputWriter!.write(buffer);
      outputWriter!.flush();
    }
  }
  
  /* finalize output */
  if outputWriter? {
    outputWriter!.endSequence();
    outputWriter!.close();
  }
}
