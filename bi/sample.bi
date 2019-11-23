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
 *
 * - `--quiet`: Don't display a progress bar.
 */
program sample(
    input:String?,
    output:String?,
    config:String?,
    seed:Integer?,
    quiet:Boolean <- false) {
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
  auto model <- Model?(make(configBuffer.getObject("model")));
  if !model? {
    error("could not create model; the model class should be given as " + 
        "model.class in the config file, and should derive from Model.");
  }

  /* sampler */
  auto sampler <- ParticleSampler?(make(configBuffer.getObject("sampler")));
  if !sampler? {
    /* revert to a default sampler */
    s:ParticleMarginalImportanceSampler;
    s.read(configBuffer.getObject("sampler"));
    sampler <- s;
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

  /* progress bar */
  bar:ProgressBar;
  if !quiet {
    bar.update(0.0);
  }

  /* sample */  
  auto f <- sampler!.sample(model!);
  auto n <- 0;
  while f? {
    sample:Model;
    lweight:Real;
    lnormalizer:Real[_];
    ess:Real[_];
    npropagations:Integer[_];
    (sample, lweight, lnormalizer, ess, npropagations) <- f!;
    
    if outputWriter? {
      buffer:MemoryBuffer;
      buffer.set("sample", sample);
      buffer.set("lweight", lweight);
      buffer.set("lnormalizer", lnormalizer);
      buffer.set("ess", ess);
      buffer.set("npropagations", npropagations);
      outputWriter!.write(buffer);
      outputWriter!.flush();
    }
          
    n <- n + 1;
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
