/**
 * Vector-bourne disease model.
 *
 *   - `T` Number of time steps.
 * 
 * Adapted from the original LibBi implementation.
 */
class VBD(T:Integer) {
  N:Integer <- 0;

  p_d_inc_h:Real;   // dealy infection by a human -> infectiousness of next human case
  p_d_inf_h:Real;   // infectious period of humans
  p_p_risk:Real;    // proportion of the population at risk
  p_p_immune:Real;  // proportion of the population at risk
  p_R0:Real;        // human-to-human basic reproduction number
  p_p_rep:Real;     // proportion of cases reported
  p_p_over:Real;    // multiplicative overdispersion of reporting
  p_s_peak:Real;    // seasonal peak week
  p_s_amp:Real;     // strength of seasonal forcing

  initI:Real;  // initial number of infectious

  S:Real[T];  // susceptible
  E:Real[T];  // incubating
  I:Real[T];  // infectious
  R:Real[T];  // recovered
  Z:Real[T];  // incidence
  beta_track:Real[T];
  reff:Real[T];

  serology_sample:Integer[T];  // input
  
  incidence:Real[T];    // observation
  serology:Integer[T];  // observation

  function input() {
    /* read in files */
  }

  function simulate() {
    /* parameters */
    p_d_inc_h <~ Gaussian(17.8/7.0, pow(2.3/7.0, 2.0)); /// @todo truncate below at 0
    p_d_inf_h <~ Gaussian(4.7/7.0, pow(1.2/7.0, 2.0));  /// @todo truncate below at 0
    p_p_immune <~ Gamma(1.0, 0.06);
    p_p_risk <~ Uniform(0.1, 1.0);
    p_R0 <~ Uniform(0.0, 25.0);
    p_s_amp <~ Uniform(0.0, 1.0);
    p_s_peak <~ Gaussian(20.0, 4.0);
    p_p_rep <~ Uniform(0.0, 1.0);
    initI <~ Gamma(1.0, 10.0);
    p_p_over <~ Beta(1.0, 10.0);
    
    /* initial conditions */
    N <- 100000;
    S[1] <- max(Real(N)*(1.0 - p_p_immune)*p_p_risk - initI, 0.0);
    E[1] <- 0.0;
    I[1] <- initI;
    R[1] <- Real(N)*p_p_immune*p_p_risk;
    Z[1] <- 0.0;
    
    beta_track[1] <- p_R0/p_d_inf_h * (1.0 + p_s_amp*cos(6.283*(-p_s_peak)/52.0));
    reff[1] <- p_R0 * S[1] / (Real(N) * p_p_risk) * (1.0 + p_s_amp*cos(6.283*(-p_s_peak)/52.0));

    incubation_rate:Real <- 1.0/p_d_inc_h;
    recovery_rate:Real <- 1.0/p_d_inf_h;
    infection_rate:Real <- p_R0/p_d_inf_h;

    t:Integer;
    for (t in 2..T) {      
      /* transition */
      transmission_rate:Real <- infection_rate*(1.0 + p_s_amp*cos(6.283*(Real(t) - p_s_peak)/52.0));
      beta_track[t] <- transmission_rate;
      reff[t] <- (transmission_rate/recovery_rate)*S[t]/(Real(N)*p_p_risk);

      transmitted:Real <- min(S[t-1], transmission_rate*S[t-1]*I[t-1]/(Real(N)*p_p_risk));
      incubated:Real <- min(E[t-1], incubation_rate*E[t-1]);
      recovered:Real <- min(I[t-1], recovery_rate*I[t-1]);
      
      S[t] <- S[t-1] - transmitted;
      E[t] <- E[t-1] + transmitted - incubated;
      I[t] <- I[t-1] + incubated - recovered;
      R[t] <- R[t-1] + recovered;
      Z[t] <- incubated;
      
      /* observe */
      if (Z[t] > 0.0) {
        incidence[t] <~ Gaussian(p_p_rep*Z[t], p_p_rep*Z[t]/(1.0 - p_p_over));  ///@todo truncate below at 0, or make Poisson?
      }
      //serology[t] <~ Binomial(serology_sample, R[t] / (N * p_p_risk)); // parameters are size and prob
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
