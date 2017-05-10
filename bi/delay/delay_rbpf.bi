cpp{{
#include <iostream>
}}

/**
 * Demonstrates a nonlinear state-space model with a linear-Gaussian
 * component that can be Rao--Blackwellised.
 */
program delay_rbpf(N:Integer <- 100, T:Integer <- 10) {  
  x:Example[N](T);
  w:Real[N];
  a:Integer[N];
  W:Real <- 0.0;
  n:Integer;
  t:Integer;

  /* initialise */
  for (n in 1..N) {
    x[n].input(T);
    x[n].parameter();
    x[n].initial();
    w[n] <- x[n].observation(1);
  }
  W <- log_sum_exp(w) - log(Real(N));
  
  /* filter */
  for (t in 2..T) {
    a <- ancestors(w);
    for (n in 1..N) {
      if (a[n] != n) {
        x[n].copy(x[a[n]], t - 1);
      }
      x[n].transition(t);
      w[n] <- x[n].observation(t);
    }
    W <- W + log_sum_exp(w) - log(Real(N));
  }
    
  /* output */
  s:Integer <- ancestor(w);
  for (t in 1..T) {
    x[s].output(t);
    print(",");
  }
  print(W);
  print("\n");
}

/**
 * The state-space model.
 */
class Example(T1:Integer) {
  T:Integer <- T1;

  Σ_x_l:Real[3,3];  // linear state noise standard deviation
  Σ_x_n:Real[1,1];  // nonlinear state noise standard deviation
  Σ_y_l:Real[1,1];  // linear observation noise standard deviation
  Σ_y_n:Real[1,1];  // nonlinear observation noise standard deviation
  
  A:Real[3,3];  // linear-linear state transition matrix
  B:Real[1,3];  // nonlinear-linear state transition matrix
  C:Real[1,3];  // linear observation matrix

  x_n:MultivariateGaussian[T](1);  // nonlinear state
  x_l:MultivariateGaussian[T](3);  // linear state
  
  y_n:MultivariateGaussian[T](1);  // nonlinear observation
  y_l:MultivariateGaussian[T](1);  // linear observation

  function parameter() {
    this.A[1,1] <- 1.0;
    this.A[1,2] <- 0.3;
    this.A[1,3] <- 0.0;
    this.A[2,1] <- 0.0;
    this.A[2,2] <- 0.92;
    this.A[2,3] <- -0.3;
    this.A[3,1] <- 0.0;
    this.A[3,2] <- 0.3;
    this.A[3,3] <- 0.92;
    
    this.B[1,1] <- 1.0;
    this.B[1,2] <- 0.0;
    this.B[1,3] <- 0.0;
    
    this.C[1,1] <- 1.0;
    this.C[1,2] <- -1.0;
    this.C[1,3] <- 1.0;
    
    this.Σ_x_l[1,1] <- 0.01;
    this.Σ_x_l[1,2] <- 0.0;
    this.Σ_x_l[1,3] <- 0.0;
    this.Σ_x_l[2,1] <- 0.0;
    this.Σ_x_l[2,2] <- 0.01;
    this.Σ_x_l[2,3] <- 0.0;
    this.Σ_x_l[3,1] <- 0.0;
    this.Σ_x_l[3,2] <- 0.0;
    this.Σ_x_l[3,3] <- 0.01;
    
    this.Σ_x_n[1,1] <- 0.01;
    this.Σ_y_l[1,1] <- 0.1;
    this.Σ_y_n[1,1] <- 0.1;
  }
  
  function initial() {
    this.x_n[1] ~ Gaussian(vector(0.0, 1), matrix(1.0, 1, 1));
    this.x_l[1] ~ Gaussian(vector(0.0, 3), identity(3, 3));
  }
  
  function transition(t:Integer) {
    this.x_n[t] ~ Gaussian(vector(atan(scalar(x_n[t-1])), 1) + B*x_l[t-1], Σ_x_n);
    this.x_l[t] ~ Gaussian(A*x_l[t-1], Σ_x_l);
  }
  
  function observation(t:Integer) -> Real {
    this.y_n[t] ~ Gaussian(vector(copysign(pow(scalar(x_n[t]), 2.0), scalar(x_n[t])), 1), Σ_y_n);
    this.y_l[t] ~ Gaussian(C*x_l[t], Σ_y_l);
    
    return y_n[t].w + y_l[t].w;
  }
  
  function input(T:Integer) {
    v:Real[T];
    t:Integer;

    v <- read("data/y_n.csv", T);
    for (t in 1..T) {
      this.y_n[t] <- v[t..t];
    }
    
    v <- read("data/y_l.csv", T);
    for (t in 1..T) {
      this.y_l[t] <- v[t..t];
    }
  }
  
  function output(t:Integer) {
    print(x_n[t]);
    print(", ");
    print(x_l[t]);
  }
    
  function copy(o:Example, t:Integer) {    
    this.Σ_x_l <- o.Σ_x_l;
    this.Σ_x_n <- o.Σ_x_n;
    this.Σ_y_l <- o.Σ_y_l;
    this.Σ_y_n <- o.Σ_y_n;
    this.A <- o.A;
    this.B <- o.B;
    this.C <- o.C;
    
    s:Integer;
    for (s in 1..t) {
      x_n[s].copy(o.x_n[s]);
      x_l[s].copy(o.x_l[s]);
      y_n[s].copy(o.y_n[s]);
      y_l[s].copy(o.y_l[s]);
    }
  }
}
