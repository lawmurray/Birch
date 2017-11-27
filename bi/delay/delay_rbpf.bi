/**
 * Demonstrates a particle filter over a nonlinear state-space model with
 * linear substructure. With delayed sampling enabled, this automatically
 * yields a Rao--Blackwellized particle filter with locally-optimal proposal.
 *
 *   - `-N`            : Number of particles.
 *   - `-T`            : Number of time steps.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 *   - `--ess-rel`     : ESS threshold, as proportion of `N`, under which
 *                       resampling is triggered.
 *
 * To disable delayed sampling, change the `~` operators to `<~` in the
 * `initial` and `transition` functions of the `Example` class, and to `~>`
 * in the `observation` function.
 */
program delay_rbpf(N:Integer <- 100, T:Integer <- 10,
    diagnostics:Boolean <- false, ess_rel:Real <- 0.7) {  
  if (diagnostics) {
    delay_rbpf_diagnostics(T);
  }

  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood
  
  /* initialize */
  for (n:Integer in 1..N) {
    x[n] <- particle(T);
  }
  W <- 0.0;
  
  for (t:Integer in 1..T) {
    /* resample */
    if (t > 1 && mod(t, 2) == 1 && ess(w) < ess_rel*N) {
      W <- W + log_sum_exp(w) - log(N);
      a <- ancestors(w);
      for (n:Integer in 1..N) {
        if (a[n] != n) {
          x[n] <- x[a[n]];
        }
        w[n] <- 0.0;
      }
    }
    
    /* propagate and weight */
    for (n:Integer in 1..N) {
      if (x[n]?) {
        w[n] <- w[n] + x[n]!;
      } else {
        w[n] <- -inf;
      }
    }
  }
  W <- W + log_sum_exp(w) - log(Real(N));
    
  /* output */
  stdout.print(N + " " + W + "\n");
}

class Example(T:Integer) {
  /**
   * Linear-linear state transition matrix.
   */
  A:Real[_,_] <- [[1.0, 0.3, 0.0], [0.0, 0.92, -0.3], [0.0, 0.3, 0.92]];
  
  /**
   * Nonlinear-linear state transition matrix.
   */
  B:Real[_,_] <- [[1.0, 0.0, 0.0]];
  
  /**
   * Linear observation matrix.
   */
  C:Real[_,_] <- [[1.0, -1.0, 1.0]];
    
  /**
   * Linear state noise covariance.
   */
  Σ_x_l:Real[_,_] <- [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
  
  /**
   * Nonlinear state noise covariance.
   */
  Σ_x_n:Real[_,_] <- [[0.01]];
  
  /**
   * Linear observation noise covariance.
   */
  Σ_y_l:Real[_,_] <- [[0.1]];
  
  /**
   * Nonlinear observation noise covariance.
   */
  Σ_y_n:Real[_,_] <- [[0.1]];

  /**
   * Nonlinear state.
   */
  x_n:Random<Real[_]>[T];
  
  /**
   * Linear state.
   */
  x_l:Random<Real[_]>[T];

  /**
   * Nonlinear observation.
   */
  y_n:Random<Real[_]>[T];
  
  /**
   * Linear observation.
   */
  y_l:Random<Real[_]>[T];

  fiber simulate() -> Real! {  
    x_n[1] ~ Gaussian(vector(0.0, 1), I(1, 1));
    x_l[1] ~ Gaussian(vector(0.0, 3), I(3, 3));
    
    y_n[1] ~ Gaussian([0.1*copysign(pow(scalar(x_n[1]), 2.0), scalar(x_n[1]))], Σ_y_n);
    y_l[1] ~ Gaussian(C*x_l[1], Σ_y_l);
    
    for (t:Integer in 2..T) {
      x_n[t] ~ Gaussian([atan(scalar(x_n[t-1]))] + B*x_l[t-1], Σ_x_n);
      x_l[t] ~ Gaussian(A*x_l[t-1], Σ_x_l);
      
      y_n[t] ~ Gaussian(vector(0.1*copysign(pow(scalar(x_n[t]), 2.0), scalar(x_n[t])), 1), Σ_y_n);
      y_l[t] ~ Gaussian(C*x_l[t], Σ_y_l);
    }
  }
  
  function input() {
    y_n_input:InputStream;
    y_n_input.open("data/y_n.csv");
    for (t:Integer in 1..T) {
      y_n[t] <- vector(y_n_input.readReal(), 1);
    }
    y_n_input.close();

    y_l_input:InputStream;
    y_l_input.open("data/y_l.csv");
    for (t:Integer in 1..T) {
      y_l[t] <- vector(y_l_input.readReal(), 1);
    }
    y_l_input.close();
  }
  
  function output() {
    for (t:Integer in 1..T) {
      stdout.print(x_n[t]);
      stdout.print(", ");
      stdout.print(x_l[t]);
      stdout.print(", ");
    }
  }
}

/*
 * Set up diagnostics.
 */
function delay_rbpf_diagnostics(T:Integer) {
  o:DelayDiagnostics(4*T);
  delayDiagnostics <- o;
  
  for (t:Integer in 1..T) {
    o.name(4*t - 3, "x_n[" + t + "]");
    o.name(4*t - 2, "x_l[" + t + "]");
    o.name(4*t - 1, "y_n[" + t + "]");
    o.name(4*t, "y_l[" + t + "]");
    
    o.position(4*t - 3, t, 4);
    o.position(4*t - 2, t, 3);
    o.position(4*t, t, 2);
    o.position(4*t - 1, t, 1);
  }
}

closed fiber particle(T:Integer) -> Real! {
  x:Example(T);
  x.input();
  x.simulate();
}
