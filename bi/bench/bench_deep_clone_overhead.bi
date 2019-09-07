/**
 * Benchmark that simulates a model while avoiding clone operations, as a
 * test of clone overhead.
 *
 * - `--config`: Name of the configuration file, if any.
 *
 * - `--diagnostic`: Name of the diagnostic file, if any.
 *
 * - `--seed`: Random number seed.
 */
program bench_deep_clone_overhead(
    config:String?,
    input:String?,
    diagnostic:String?,
    output:String?,
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
  modelClass:String?;  
  if config? {
    auto buffer <- configBuffer.getObject("model");
    if buffer? {
      modelClass <- buffer!.getString("class");
    }
  }
  if !modelClass? {
    error("no model class specified, this should be given using the --model option, or as model.class in the config file.");
  }
  
  /* input */
  inputBuffer:MemoryBuffer;
  if input? {
    reader:Reader <- Reader(input!);
    reader.read(inputBuffer);    
    reader.close();
  }
  
  /* sampler */
  nsamples:Integer? <- 1;
  nparticles:Integer? <- 1;
  nsteps:Integer? <- 1;
  if config? {
    auto buffer <- configBuffer.getObject("sampler");
    if buffer? {
      nsamples <- buffer!.getInteger("nsamples");
      nparticles <- buffer!.getInteger("nparticles");
      nsteps <- buffer!.getInteger("nsteps");
    }
  }
  if !nsamples? {
    nsamples <- 1;
  }
  if !nparticles? {
    nparticles <- 1;
  }
  if !nsteps? {
    nsteps <- 0;
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

  /* simulate */
  x:ForwardModel[nparticles!];
  for auto m in 1..nsamples! {
    memory:Integer <- 0;
    elapsed:Real <- 0.0;
    tic();
    
    /* initialize */
    for auto n in 1..nparticles! {
      auto m <- ForwardModel?(make(modelClass!));    
      if m? {
        if config? {
          configBuffer.get("model", m!);
        }
        if input? {
          inputBuffer.get(m!);
        }
        x[n] <- m!;
        x[n].start();
      } else {
        error(modelClass! + " is not a subtype of Model.");
      }
    }
    
    /* transition */
    for auto t in 1..nsteps! {
      parallel for auto n in 1..nparticles! {
        x[n].step();
      }
      memory <- max(memory, memoryUse());
    }
    elapsed <- toc();
    
    buffer:MemoryBuffer;
    buffer.set("memory", memory);
    buffer.set("elapsed", elapsed);
    diagnosticWriter!.write(buffer);
    diagnosticWriter!.flush();
  }

  if diagnosticWriter? {
    diagnosticWriter!.endSequence();
    diagnosticWriter!.close();
  }
}
