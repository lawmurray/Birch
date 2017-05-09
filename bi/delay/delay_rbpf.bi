/**
 * Demonstrates a nonlinear state-space model with a linear-Gaussian
 * component that can be Rao--Blackwellised.
 */
program delay_rbpf(T:Integer <- 10) {
  m:Example(T);
  t:Integer;
  
  m.parameter();
  m.initial();
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
  L_x_l:Real[3,3];     // linear state noise standard deviation
  L_x_n:Real[1,1];     // nonlinear state noise standard deviation
  L_y_l:Real[1,1];     // linear observation noise standard deviation
  L_y_n:Real[1,1];     // nonlinear observation noise standard deviation
  
  A:Real[3,3];  // linear-linear state transition matrix
  B:Real[1,3];  // nonlinear-linear state transition matrix
  C:Real[1,3];  // linear observation matrix

  x_n:Real[T,1];    // nonlinear state
  x_l:Real[T,3];  // linear state
  
  y_n:Real[T,1];    // nonlinear observation
  y_l:Real[T,1];  // linear observation

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
    
    this.L_x_l[1,1] <- 0.01;
    this.L_x_l[1,2] <- 0.0;
    this.L_x_l[1,3] <- 0.0;
    this.L_x_l[2,1] <- 0.0;
    this.L_x_l[2,2] <- 0.01;
    this.L_x_l[2,3] <- 0.0;
    this.L_x_l[3,1] <- 0.0;
    this.L_x_l[3,2] <- 0.0;
    this.L_x_l[3,3] <- 0.01;
    
    this.L_x_n[1,1] <- 0.01;
    this.L_y_l[1,1] <- 0.1;
    this.L_y_n[1,1] <- 0.1;
  }
  
  function initial() {
    this.x_n[1,1] <~ Gaussian(0.0, 1.0);
    i:Integer;
    for (i in 1..3) {
      this.x_l[1,i] <- 0.0;
    }
  }
  
  function transition(t:Integer) {
    this.x_n[t,1..1] <~ Gaussian(vector(atan(x_n[t-1,1]), 1) + B*x_l[t-1,1..3], L_x_n);
    this.x_l[t,1..3] <~ Gaussian(A*x_l[t-1,1..3], L_x_l);
  }
  
  function observation(t:Integer) -> Real {
    this.y_n[t,1..1] <~ Gaussian(vector(copysign(pow(x_n[t,1], 2.0), x_n[t,11]), 1), L_y_n);
    this.y_l[t,1..1] <~ Gaussian(C*x_l[t,1..3], L_y_l);
  }
  
  function output(t:Integer) {
    print(x_n[t,1]);
    print("\t");
    print(x_l[t,1]);
    print("\t");
    print(x_l[t,2]);
    print("\t");
    print(x_l[t,3]);
    print("\t");
    print(y_n[t,1]);
    print("\t");
    print(y_l[t,1]);
    print("\n");
  }
}
