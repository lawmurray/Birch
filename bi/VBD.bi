/**
 * Vector-bourne disease model.
 *
 *   - `T` Number of time steps.
 * 
 * Adapted from the original LibBi implementation.
 */
class VBD(T:Integer) {
  N:Integer <- 0;

  p_d_inc_h:Gaussian;   // dealy infection by a human -> infectiousness of next human case
  p_d_inf_h:Gaussian;   // infectious period of humans
  p_p_risk:Uniform;    // proportion of the population at risk
  p_p_immune:Gamma;  // proportion of the population at risk
  p_R0:Uniform;        // human-to-human basic reproduction number
  p_s_amp:Uniform;     // strength of seasonal forcing
  p_s_peak:Gaussian;    // seasonal peak week
  p_p_rep:Uniform;     // proportion of cases reported
  p_p_over:Beta;    // multiplicative overdispersion of reporting

  λ:Binomial[T];  // newly infected population
  δ:Binomial[T];  // newly incubated population
  γ:Binomial[T];  // newly recovered population

  S:Real[T];  // susceptible population
  E:Real[T];  // incubating population
  I:Real[T];  // infectious population
  R:Real[T];  // recovered population

  serology_sample:Integer[T];  // input
  incidence:Gaussian[T];    // observation
  serology:Binomial[T];  // observation

  function input() {
    /* read in files */
  }

  function simulate() {
    /* parameters */
    p_d_inc_h ~ Gaussian(17.8/7.0, pow(2.3/7.0, 2.0)); /// @todo truncate below at 0
    p_d_inf_h ~ Gaussian(4.7/7.0, pow(1.2/7.0, 2.0));  /// @todo truncate below at 0
    p_p_risk ~ Uniform(0.1, 1.0);
    p_p_immune ~ Gamma(1.0, 0.06);
    p_R0 ~ Uniform(0.0, 25.0);
    p_s_amp ~ Uniform(0.0, 1.0);
    p_s_peak ~ Gaussian(20.0, 4.0);
    p_p_rep ~ Uniform(0.0, 1.0);
    p_p_over ~ Beta(1.0, 10.0);
    
    /* initial conditions */
    N <- 100000;
    
    E[1] <- 0.0;
    I[1] <~ Gamma(1.0, 10.0);
    R[1] <- Real(N)*p_p_immune*p_p_risk;
    S[1] <- max(Real(N) - I[1] - R[1], 0.0);
    
    for (t:Integer in 2..T) {
      /* transition */
      incubation_rate:Real <- 1.0/p_d_inc_h;
      recovery_rate:Real <- 1.0/p_d_inf_h;
      infection_rate:Real <- p_R0.value()/p_d_inf_h;
      transmission_rate:Real <- infection_rate*(1.0 + p_s_amp*cos(2.0*π*(Real(t) - p_s_peak)/52.0));

      λ[t] ~ Binomial(Integer(S[t-1]*I[t-1]/(Real(N)*p_p_risk)), transmission_rate);
      δ[t] <- E[t-1]*incubation_rate;
      γ[t] <- I[t-1]*recovery_rate;
      
      S[t] <- S[t-1] - Real(λ[t]);
      E[t] <- E[t-1] + Real(λ[t]) - Real(δ[t]);
      I[t] <- I[t-1] + Real(δ[t]) - Real(γ[t]);
      R[t] <- R[t-1] + Real(γ[t]);
      
      /* observe */
      incidence[t] ~ Gaussian(p_p_rep*Real(δ[t]), p_p_rep*Real(δ[t])/(1.0 - p_p_over));  ///@todo truncate below at 0, or make Poisson?
      // ^ use a negative binomial here, as an overdispersed Poisson?
      serology[t] ~ Binomial(serology_sample[t], Real(R[t]) / (Real(N) * p_p_risk)); // parameters are size and prob
      // ^ can probability be beta distributed?
    }
  }

  function output() {
    /* output a sample */
    t:Integer;
    for (t in 1..T) {
      stdout.print(S[t] + "," + E[t] + "," + I[t] + "," + R[t] + "\n");
    }
  }
}
