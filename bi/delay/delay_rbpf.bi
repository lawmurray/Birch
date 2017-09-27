/**
 * Demonstrates a particle filter over a nonlinear state-space model with
 * linear substructure. With delayed sampling enabled, this automatically
 * yields a Rao--Blackwellized particle filter with locally-optimal proposal.
 *
 *   - `-N`            : Number of particles.
 *   - `-T`            : Number of time steps.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 *
 * To disable delayed sampling, change the `~` operators to `<~` in the
 * `initial` and `transition` functions of the `Example` class, and to `~>`
 * in the `observation` function.
 */
program delay_rbpf(N:Integer <- 100, T:Integer <- 10,
    diagnostics:Boolean <- false) {  
  if (diagnostics) {
    delay_rbpf_diagnostics(T);
  }

  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood
  
  /* initialize */
  p:Real! <- particle(T);
  for (n:Integer in 1..N) {
    x[n] <- p;
  }
  W <- 0.0;
  
  for (t:Integer in 1..T) {
    /* resample */
    if (t > 1 && mod(t, 2) == 1) {
      a <- ancestors(w);
      for (n:Integer in 1..N) {
        if (a[n] != n) {
          x[n] <- x[a[n]];
        }
      }
    }
    
    /* propagate and weight */
    for (n:Integer in 1..N) {
      if (x[n]?) {
        w[n] <- x[n]!;
      } else {
        w[n] <- -inf;
      }
    }
    
    /* marginal log-likelihood estimate */
    W <- W + log_sum_exp(w) - log(Real(N));
  }
    
  /* output */
  stdout.print(W + "\n");
}

fiber particle(T:Integer) -> Real! {
  x:Example(T);
  
  x.input();
  x.simulate();
}

class Example(T:Integer) {
  Σ_x_l:Real[3,3];  // linear state noise covariance
  Σ_x_n:Real[1,1];  // nonlinear state noise covariance
  Σ_y_l:Real[1,1];  // linear observation noise covariance
  Σ_y_n:Real[1,1];  // nonlinear observation noise covariance
  
  A:Real[3,3];  // linear-linear state transition matrix
  B:Real[1,3];  // nonlinear-linear state transition matrix
  C:Real[1,3];  // linear observation matrix

  x_n:MultivariateGaussian[T](1);  // nonlinear state
  x_l:MultivariateGaussian[T](3);  // linear state
  
  y_n:MultivariateGaussian[T](1);  // nonlinear observation
  y_l:MultivariateGaussian[T](1);  // linear observation

  fiber simulate() -> Real! {
    A[1,1] <- 1.0;
    A[1,2] <- 0.3;
    A[1,3] <- 0.0;
    A[2,1] <- 0.0;
    A[2,2] <- 0.92;
    A[2,3] <- -0.3;
    A[3,1] <- 0.0;
    A[3,2] <- 0.3;
    A[3,3] <- 0.92;
    
    B[1,1] <- 1.0;
    B[1,2] <- 0.0;
    B[1,3] <- 0.0;
    
    C[1,1] <- 1.0;
    C[1,2] <- -1.0;
    C[1,3] <- 1.0;
    
    Σ_x_l[1,1] <- 0.01;
    Σ_x_l[1,2] <- 0.0;
    Σ_x_l[1,3] <- 0.0;
    Σ_x_l[2,1] <- 0.0;
    Σ_x_l[2,2] <- 0.01;
    Σ_x_l[2,3] <- 0.0;
    Σ_x_l[3,1] <- 0.0;
    Σ_x_l[3,2] <- 0.0;
    Σ_x_l[3,3] <- 0.01;
    
    Σ_x_n[1,1] <- 0.01;
    Σ_y_l[1,1] <- 0.1;
    Σ_y_n[1,1] <- 0.1;

    x_n[1] ~ Gaussian(vector(0.0, 1), I(1, 1));
    x_l[1] ~ Gaussian(vector(0.0, 3), I(3, 3));

    y_n[1] ~ Gaussian(vector(0.1*copysign(pow(scalar(x_n[1]), 2.0), scalar(x_n[1])), 1), Σ_y_n);
    y_l[1] ~ Gaussian(C*x_l[1], Σ_y_l);

    for (t:Integer in 2..T) {
      x_n[t] ~ Gaussian(vector(atan(scalar(x_n[t-1])), 1) + B*x_l[t-1], Σ_x_n);
      x_l[t] ~ Gaussian(A*x_l[t-1], Σ_x_l);

      y_n[t] ~ Gaussian(vector(0.1*copysign(pow(scalar(x_n[t]), 2.0), scalar(x_n[t])), 1), Σ_y_n);
      y_l[t] ~ Gaussian(C*x_l[t], Σ_y_l);
    }
  }
  
  function input() {
    v:Real[T];

    v <- read("data/y_n.csv", T);
    for (t:Integer in 1..T) {
      y_n[t] <- v[t..t];
    }
    
    v <- read("data/y_l.csv", T);
    for (t:Integer in 1..T) {
      y_l[t] <- v[t..t];
    }
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
