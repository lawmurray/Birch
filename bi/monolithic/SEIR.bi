/**
 * SEIR (susceptible-exposed-infectious-recovered) model with discrete states
 * and binomial transfer.
 *
 *   - `T` Number of time steps.
 */
class SEIR(T:Integer) {
  ν:Beta;   // birth probability
  μ:Beta;   // survival probability
  λ:Beta;   // exposure probability
  δ:Beta;   // infection probability
  γ:Beta;   // recovery probability

  s:Integer[T];    // susceptible population
  e:Integer[T];    // incubating population
  i:Integer[T];    // infectious population
  r:Integer[T];    // recovered population

  Δs:Integer[T];   // newly susceptible (births)
  Δe:Integer[T];   // newly exposed
  Δi:Integer[T];   // newly infected
  Δr:Integer[T];   // newly recovered

  n:Integer[T];    // total population

  fiber simulate() -> Real! {
    /* parameters */
    ν <- 0.0;
    μ <- 1.0;
    λ ~ Beta(1.0, 1.0);
    δ ~ Beta(1.0, 1.0);
    γ ~ Beta(1.0, 1.0);
  
    /* initial conditions */
    s[1] <- 10000;
    e[1] <- 0;
    i[1] <- 1;
    r[1] <- 0;

    Δs[1] <- 0;
    Δe[1] <- 0;
    Δi[1] <- 1;
    Δr[1] <- 0;
    
    n[1] <- s[1] + e[1] + i[1] + r[1];

    /* transition */
    for (t:Integer in 2..T) {
      /* transfers */
      Δe[t] <~ Binomial(s[t-1]*i[t-1]/n[t-1], λ);
      Δi[t] <~ Binomial(e[t-1], δ);
      Δr[t] <~ Binomial(i[t-1], γ);

      s[t] <- s[t-1] - Δe[t];
      e[t] <- e[t-1] + Δe[t] - Δi[t];
      i[t] <- i[t-1] + Δi[t] - Δr[t];
      r[t] <- r[t-1] + Δr[t];
    
      /* deaths */
      s[t] <~ Binomial(s[t], μ);
      e[t] <~ Binomial(e[t], μ);
      i[t] <~ Binomial(i[t], μ);
      r[t] <~ Binomial(r[t], μ);

      /* births */
      Δs[t] <~ Binomial(n[t-1], ν);
      s[t] <- s[t] + Δs[t];
    
      /* total population */
      n[t] <- s[t] + e[t] + i[t] + r[t];
    }
  }
}
