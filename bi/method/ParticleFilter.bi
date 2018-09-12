/**
 * Particle filter.
 */
class ParticleFilter {
  /**
   * Canonical particle from which others are initialized. Ensures that
   * input is only consumed once.
   */
  f0:WeightedVariate<Object>!;

  /**
   * Particles.
   */
  f:WeightedVariate<Object>![_];
  
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
   * For each checkpoint, the normalizing constant estimate up to that
   * checkpoint.
   */
  Z:Real[_];
  
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
   * - M: Number of samples.
   * - T: Number of checkpoints.
   * - N: Number of particles.
   * - trigger: Relative ESS below which resampling should be triggered.
   */
  function sample(model:String, inputReader:Reader?, outputWriter:Writer?,
      diagnosticWriter:Writer?, M:Integer, T:Integer, N:Integer,
      trigger:Real, verbose:Boolean) {
    /* set up output */
    if (M > 1) {
      if (outputWriter?) {
        outputWriter!.setArray();
      }
      if (diagnosticWriter?) {
        diagnosticWriter!.setArray();
      }
    }

    initialize(model, inputReader, T, N, trigger);    
    for (m:Integer in 1..M) {
      start();
      for (t:Integer in 1..T) {
        if (T > 1 && verbose) {
          stderr.print(t + " ");
        }
        step(t);
      }
      if (verbose) {
        stderr.print(Z[T] + "\n");
      }
      finish();
            
      /* output results and diagnostics */
      if (outputWriter?) {
        if (M > 1) {
          output(outputWriter!.push());
        } else {
          output(outputWriter!);
        }
      }
      if (diagnosticWriter?) {
        if (M > 1) {
          diagnose(diagnosticWriter!.push());
        } else {
          diagnose(diagnosticWriter!);
        }
      }
    }
  }

  /**
   * Initialize the method.
   *
   * - T: Number of checkpoints.
   * - N: Number of particles.
   * - trigger: Relative ESS below which resampling should be triggered.
   */
  function initialize(model:String, reader:Reader?, T:Integer, N:Integer, trigger:Real) {
    /* model */
    auto o <- AbstractModel?(make(model));
    if (!o?) {
      stderr.print("error: " + model + " must be a subtype of AbstractModel with no initialization parameters.\n");
      exit(1);
    }
    auto m <- o!;
  
    /* variate */
    v:WeightedVariate<Object>(m.variate(), 0.0);
  
    /* input */
    if (reader?) {
      v.x.read(reader!);
    }
  
    f0 <- particle(m, v);
    f1:WeightedVariate<Object>![N];
    this.f <- f1;
    this.T <- T;
    this.N <- N;
    this.trigger <- trigger;
  }

  /**
   * Start the filter.
   *
   * - model: Name of the model class.
   * - inputReader: Reader for input.
   */  
  function start() {
    for (n:Integer in 1..N) {
      f[n] <- f0;
    }
    this.w <- vector(0.0, N);
    this.e <- vector(0.0, T);
    this.r <- vector(false, T);
    this.Z <- vector(0.0, T);
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
      stderr.print("error: particle filter degenerated.\n");
      exit(1);
    }
    r[t] <- e[t] < trigger*N;
    if (r[t]) {
      auto a <- ancestors(w);
      auto g <- f;
      for (n:Integer in 1..N) {
        f[n] <- g[a[n]];
        w[n] <- 0.0;
      }
    }

    /* propagate and weight */
    parallel for (n:Integer in 1..N) {
      if (f[n]?) {
        w[n] <- w[n] + f[n]!.w;
      } else {
        stderr.print("error: particles terminated prematurely.\n");
        exit(1);
      }
    }
    
    /* update normalizing constant estimate */
    auto W <- log_sum_exp(w);
    w <- w - (W - log(N));
    if (t > 1) {
      Z[t] <- Z[t - 1] + (W - log(N));
    } else {
      Z[t] <- W - log(N);
    }
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
      auto b <- ancestor(w);
      if (b > 0) {
        sample:WeightedVariate<Object>(f[b]!.x, Z[T]);
        writer!.write(sample);
      } else {
        stderr.print("error: particle filter degenerated.\n");
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
      writer!.setRealVector("ess", e);
      writer!.setBooleanVector("resample", r);
      writer!.setRealVector("evidence", Z);
    }
  }
}

/*
 * Particle.
 */
fiber particle(m:AbstractModel, v:WeightedVariate<Object>) -> WeightedVariate<Object> {
  auto f <- m.simulate(v.x);
  while (f?) {
    v.w <- f!;
    yield v;
  }
}
