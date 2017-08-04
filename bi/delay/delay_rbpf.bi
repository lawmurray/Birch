/**
 * The state-space model.
 */
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
    t:Integer <- 1;
    input();
    parameter();
    initial();
    yield observation(t);
    for (t in 2..T) {
      transition(t);
      yield observation(t);
    }
  }

  function parameter() {
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
  }
  
  function initial() {
    x_n[1] ~ Gaussian(vector(0.0, 1), identity(1, 1));
    x_l[1] ~ Gaussian(vector(0.0, 3), identity(3, 3));
  }
  
  function transition(t:Integer) {
    x_n[t] ~ Gaussian(vector(atan(scalar(x_n[t-1])), 1) + B*x_l[t-1], Σ_x_n);
    x_l[t] ~ Gaussian(A*x_l[t-1], Σ_x_l);
  }
  
  function observation(t:Integer) -> Real {
    y_n[t] ~ Gaussian(vector(0.1*copysign(pow(scalar(x_n[t]), 2.0), scalar(x_n[t])), 1), Σ_y_n);
    y_l[t] ~ Gaussian(C*x_l[t], Σ_y_l);
    
    return y_n[t].w + y_l[t].w;
  }
  
  function input() {
    v:Real[T];
    t:Integer;

    v <- read("data/y_n.csv", T);
    for (t in 1..T) {
      y_n[t] <- v[t..t];
    }
    
    v <- read("data/y_l.csv", T);
    for (t in 1..T) {
      y_l[t] <- v[t..t];
    }
  }
  
  function output() {
    t:Integer;
    for (t in 1..T) {
      print(x_n[t]);
      print(", ");
      print(x_l[t]);
      print(", ");
    }
  }
}

fiber particle(T:Integer) -> Real! {
  x:Example(T);
  w:Real;
  f:Real! <- x.simulate();
  while (f?) {
    yield f!;  
  }
}

/**
 * Demonstrates a particle filter over a nonlinear state-space model with
 * linear substructure. With delayed sampling enabled, this automatically
 * yields a Rao--Blackwellized particle filter with locally-optimal proposal.
 *
 * `N` Number of particles.
 * `T` Number of time steps.
 *
 * To disable delayed sampling, change the `~` operators to `<~` in the
 * `initial` and `transition` functions of the `Example` class.
 */
program delay_rbpf(N:Integer <- 100, T:Integer <- 10) {  
  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood
  
  n:Integer;
  t:Integer <- 1;

  /* initialize */
  for (n in 1..N) {
    x[n] <- particle(T);
    if (x[n]?) {
      w[n] <- x[n]!;
    } else {
      w[n] <- -inf;
    }
  }
  W <- log_sum_exp(w) - log(Real(N));
  
  for (t in 2..T) {
    /* resample */
    a <- ancestors(w);
    for (n in 1..N) {
      if (a[n] != n) {
        x[n] <- x[a[n]];
      }
    }
    
    /* propagate and weight */
    for (n in 1..N) {
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
  //x[ancestor(w)].output();
  print(W);
  print(",");
  print(N);
  print("\n");
}
