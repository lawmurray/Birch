/**
 * Filter a model.
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
program filter(
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

  /* filter */
  auto filter <- ParticleFilter?(make(configBuffer.getObject("filter")));
  if !filter? {
    /* revert to a default filter */
    f:ParticleFilter;
    f.read(configBuffer.getObject("filter"));
	filter <- f;
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

  /* filter */
  auto f <- filter!.filter(model!);
  while f? {    
    if outputWriter? {
      sample:Model[_];
      lweight:Real[_];
      lnormalizer:Real;
      ess:Real;
      propagations:Integer;
      (sample, lweight, lnormalizer, ess, propagations) <- f!;

      buffer:MemoryBuffer;
	  buffer.set("sample", sample);
	  buffer.set("lweight", lweight);
	  buffer.set("lnormalizer", lnormalizer);
	  buffer.set("ess", ess);
	  buffer.set("npropagations", propagations);

      /* forecast */
	  auto forecast <- buffer.setArray("forecast");
	  auto g <- filter!.forecast(sample, lweight);
	  while g? {
	    auto buffer <- forecast.push();
	    (sample, lweight) <- g!;
        buffer.set("sample", sample);
	    buffer.set("lweight", lweight);
	  }

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
