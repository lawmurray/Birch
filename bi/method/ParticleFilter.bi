/**
 * Particle filter.
 */
class ParticleFilter {
  /**
   * Particles.
   */
  f:Model![_];
  
  /**
   * Log-weights.
   */
  w:Real[_];
  
  /**
   * For each checkpoint, the effective sample size (ESS).
   */
  e:Real[_];
  
  /**
   * For each checkpoint, was resampling performed?
   */
  r:Boolean[_];
  
  /**
   * Normalizing constant estimate.
   */
  Z:Real;
  
  /**
   * Number of checkpoints.
   */
  T:Integer;
  
  /**
   * Number of particles.
   */
  N:Integer;
  
  /**
   * Relative ESS below which resampling should be triggered.
   */
  trigger:Real;

  /**
   * Run the filter.
   *
   * - model: Name of the model class.
   * - inputReader: Reader for input.
   * - outputWriter: Writer for output.
   * - diagnosticWriter: Writer for diagnostics.
   * - T: Number of checkpoints.
   * - N: Number of particles.
   * - trigger: Relative ESS below which resampling should be triggered.
   */
  function filter(model:String, inputReader:Reader?, outputWriter:Writer?,
      diagnosticWriter:Writer?, T:Integer, N:Integer, trigger:Real) {
    initialize(T, N, trigger);
    start(model, inputReader);
    for (t:Integer in 1..T) {
      stderr.print(t + " ");
      step(t);
    }
    finish();
    stderr.print(Z + "\n");    
    output(outputWriter);
    diagnose(diagnosticWriter);
  }

  /**
   * Initialize the method.
   *
   * - T: Number of checkpoints.
   * - N: Number of particles.
   * - trigger: Relative ESS below which resampling should be triggered.
   */
  function initialize(T:Integer, N:Integer, trigger:Real) {
    f1:Model![N];
    this.f <- f1;
    this.w <- vector(0.0, N);
    this.e <- vector(0.0, T);
    this.r <- vector(false, T);
    this.T <- T;
    this.N <- N;
    this.trigger <- trigger;
    this.Z <- 0.0;
  }

  /**
   * Start the filter.
   *
   * - model: Name of the model class.
   * - inputReader: Reader for input.
   */  
  function start(model:String, inputReader:Reader?) {
    /* run one particle to its first checkpoint, which occurs immediately
     * after the input file has been read, then copy it around */     
    f0:Model! <- particle(model, inputReader);
    if (f0?) {
      for (n:Integer in 1..N) {
        f[n] <- f0;
      }
    } else {
      stderr.print("error: particles terminated prematurely.\n");
      exit(1);
    }
  }
  
  /**
   * Step to the next checkpoint.
   *
   * - t: Checkpoint number.
   */
  function step(t:Integer) {
    /* resample (if necessary) */
    e[t] <- ess(w);
    if (!(e[t] > 0.0)) {  // may be nan
      stderr.print("error: filter degenerated.\n");
      exit(1);
    }
    r[t] <- e[t] < trigger*N;
    if (r[t]) {
      a:Integer[_] <- permute_ancestors(ancestors(w));
      for (n:Integer in 1..N) {
        f[n] <- f[a[n]];
        w[n] <- -log(N);
      }
    }

    /* propagate and weight */
    for (n:Integer in 1..N) {
      if (f[n]?) {
        w[n] <- w[n] + f[n]!.w;
      } else {
        stderr.print("error: particles terminated prematurely.\n");
        exit(1);
      } 
    }
    
    /* update normalizing constant estimate */
    W:Real <- log_sum_exp(w);
    w <- w - (W - log(N));
    Z <- Z + (W - log(N));
  }
  
  /**
   * Finish the filter.
   */
  function finish() {
    //
  }
  
  /**
   * Write output.
   *
   * - outputWriter: Writer for output.
   */
  function output(writer:Writer?) {
    if (writer?) {
      b:Integer <- ancestor(w);
      if (b > 0) {
        f[b]!.output(writer!);
      } else {
        stderr.print("error: filter degenerated.\n");
        exit(1);
      }
    }
  }
  
  /**
   * Write diagnostics.
   *
   * - diagnosticWriter: Writer for diagnostics.
   */
  function diagnose(writer:Writer?) {
    if (writer?) {
      writer!.setRealArray("ess", e);
      writer!.setBooleanArray("resample", r);
    }
  }
}

/*
 * Particle.
 */
fiber particle(model:String, reader:Reader?) -> Model! {
  /* create model */
  x:Model? <- Model?(make(model));
  if (!x?) {
    stderr.print("error: " + model + " must be a subtype of Model with no initialization parameters.\n");
    exit(1);
  }
  
  /* input */
  if (reader?) {
    x!.input(reader!);
  }
  yield x!;
  
  /* simulate */
  f:Real! <- x!.simulate();
  while (f?) {
    x!.w <- f!;
    yield x!;
  }
}
