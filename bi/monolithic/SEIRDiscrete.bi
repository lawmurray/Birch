/**
 * SEIR (susceptible-exposed-infectious-recovered) model with discrete states
 * and binomial transfer.
 *
 *   - `T` Number of time steps.
 */
class SEIRDiscrete(T:Integer) {
  ρ_λ:Beta;  // exposure probability
  ρ_δ:Beta;  // infection probability
  ρ_γ:Beta;  // recovery probability
  ρ_y:Beta;  // reporting probability

  S:Integer[T];  // susceptible population
  E:Integer[T];  // incubating population
  I:Integer[T];  // infectious population
  R:Integer[T];  // recovered population
  N:Integer;     // total population

  λ:Binomial[T];  // newly infected population
  δ:Binomial[T];  // newly incubated population
  γ:Binomial[T];  // newly recovered population

  y_δ:Binomial[T];  // observed number of newly infected
  y_N:Integer[T];   // number of serology samples
  y_R:Binomial[T];  // number of positive serology samples

  fiber run() -> Real! {
    ρ_λ ~ Beta(1.0, 1.0);
    ρ_δ ~ Beta(7.0/17.8, 1.0 - 7.0/17.8);
    ρ_γ ~ Beta(7.0/4.7, 1.0 - 7.0/4.7);    
    ρ_y ~ Beta(1.0, 1.0);
        
    N <- 100000;
    R[1] <- Integer(N*0.06);
    I[1] <- 10;
    E[1] <- 0;
    S[1] <- N - I[1] - R[1] - E[1];
    
    for (t:Integer in 2..T) {
      λ[t] ~ Binomial(S[t-1]*I[t-1]/N, ρ_λ);
      δ[t] ~ Binomial(E[t-1], ρ_δ);
      γ[t] ~ Binomial(I[t-1], ρ_γ);

      y_δ[t] ~ Binomial(δ[t], ρ_y);

      S[t] <- S[t-1] - λ[t];
      E[t] <- E[t-1] + λ[t] - δ[t];
      I[t] <- I[t-1] + δ[t] - γ[t];
      R[t] <- R[t-1] + γ[t];
      
      y_R[t] ~ Binomial(y_N[t], Real(R[t])/Real(N));
    }
  }

  function input() {
    // read in files
  }

  function output() {
    t:Integer;
    for (t in 1..T) {
      stdout.print(S[t] + ",");
      stdout.print(E[t] + ",");
      stdout.print(I[t] + ",");
      stdout.print(R[t] + ",");
      stdout.print(λ[t] + ",");
      stdout.print(δ[t] + ",");
      stdout.print(γ[t] + "\n");
    }
  }
}
