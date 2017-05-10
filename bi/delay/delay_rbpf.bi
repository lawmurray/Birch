cpp{{
#include <iostream>
}}

/**
 * Demonstrates a nonlinear state-space model with a linear-Gaussian
 * component that can be Rao--Blackwellised.
 */
program delay_rbpf(T:Integer <- 10) {
  m:Example(T);
  t:Integer;
  
  m.input(T);
  m.parameter();
  m.initial();
  m.observation(1);
  for (t in 2..T) {
    m.transition(t);
    m.observation(t);
  }
  for (t in 1..T) {
    m.output(t);
  }
}

/**
 * The state-space model.
 */
class Example(T:Integer) {
  Σ_x_l:Real[3,3];     // linear state noise standard deviation
  Σ_x_n:Real[1,1];     // nonlinear state noise standard deviation
  Σ_y_l:Real[1,1];     // linear observation noise standard deviation
  Σ_y_n:Real[1,1];     // nonlinear observation noise standard deviation
  
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
    //this.x_l[1] <- vector(0.0, 3);
    this.x_l[1] ~ Gaussian(vector(0.0, 3), identity(3, 3));
  }
  
  function transition(t:Integer) {
    this.x_n[t] ~ Gaussian(vector(atan(scalar(x_n[t-1])), 1) + B*x_l[t-1], Σ_x_n);
    this.x_l[t] ~ Gaussian(A*x_l[t-1], Σ_x_l);
  }
  
  function observation(t:Integer) {
    this.y_n[t] ~ Gaussian(vector(copysign(pow(scalar(x_n[t]), 2.0), scalar(x_n[t])), 1), Σ_y_n);
    this.y_l[t] ~ Gaussian(C*x_l[t], Σ_y_l);
  }
  
  function input(T:Integer) {
    v:Real[T];
    t:Integer;

    v <- read("data/y_n.csv", T);
    for (t in 1..T) {
      this.y_n[t] <- vector(v[t], 1);
    }
    
    v <- read("data/y_l.csv", T);
    for (t in 1..T) {
      this.y_l[t] <- vector(v[t], 1);
    }
  }
  
  function output(t:Integer) {
    print(x_n[t]);
    print(", ");
    print(x_l[t]);
    print(", ");
    print(y_n[t]);
    print(", ");
    print(y_l[t]);
    print(", ");
    print(y_n[t].w + y_l[t].w);
    print("\n");
  }
}
