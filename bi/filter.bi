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
 * - `--model`: Name of the model class, if any. Alternatively, provide this
 *   as `model.class` in the configuration file.
 *
 * - `--seed`: Random number seed. Alternatively, provide this as `seed` in
 *   the configuration file. If not provided, random entropy is used.
 *
 * - `--quiet`: Don't display a progress bar.
 */
program filter(
    input:String?,
    output:String?,
    config:String?,
    model:String?,
    seed:Integer?,
    quiet:Boolean) {
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
  auto buffer <- configBuffer.getObject("model");
  if !buffer? {
    buffer <- configBuffer.setObject("model");
  }
  if !buffer!.getString("class")? && model? {
    buffer!.setString("class", model!);
  }
  auto m <- Model?(make(buffer));
  if !m? {
    error("could not create model; the model class should be given as " + 
        "model.class in the config file, and should derive from Model.");
  }

  /* filter */
  buffer <- configBuffer.getObject("filter");
  if !buffer? {
    buffer <- configBuffer.setObject("filter");
  }
  if !buffer!.getString("class")? {
    buffer!.setString("class", "ParticleFilter");
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
    reader:Reader <- Reader(inputPath!);
    inputBuffer:MemoryBuffer;
    reader.read(inputBuffer);
    reader.close();
    inputBuffer.get(m!);
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

  /* filter */
  auto f <- filter!.filter(m!);
  auto t <- 0;
  while f? {
    sample:Model[_];
    lweight:Real[_];
    lnormalize:Real;
    ess:Real;
    propagations:Integer;
    (sample, lweight, lnormalize, ess, propagations) <- f!;

    /* write filter distribution to buffer */
    buffer:MemoryBuffer;
    if outputWriter? {
      buffer.set("sample", sample);
      buffer.set("lweight", lweight);
      buffer.set("lnormalize", lnormalize);
      buffer.set("ess", ess);
      buffer.set("npropagations", propagations);
    }
    
    /* forecast */
    auto forecast <- buffer.setArray("forecast");
    auto g <- filter!.forecast(t, sample, lweight);
    while g? {
      (sample, lweight) <- g!;
      
      /* write forecast to buffer */
      if outputWriter? {
        auto buffer <- forecast.push();
        buffer.set("sample", sample);
        buffer.set("lweight", lweight);
      }
    }

    /* write buffer to file */
    if outputWriter? {
      outputWriter!.write(buffer);
      outputWriter!.flush();
    }
    
    t <- t + 1;
    if !quiet {
      bar.update(Real(t)/(filter!.nsteps! + 1));
    }
  }
  
  /* finalize output */
  if outputWriter? {
    outputWriter!.endSequence();
    outputWriter!.close();
  }
}
