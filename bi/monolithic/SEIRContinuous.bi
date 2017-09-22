/**
 * SEIR (susceptible-exposed-infectious-recovered) model with continuous
 * states and log-normal transfer.
 *
 *   - `T` Number of time steps.
 */
class SEIRContinuous(T:Integer) {
  d_inc_h:Real;   // delay infection by a human -> infectiousness of next human case
  d_inf_h:Real;   // infectious period of humans
  p_risk:Real;    // proportion of the population at risk
  p_immune:Real;  // proportion of the population at risk
  R0:Real;        // human-to-human basic reproduction number
  s_amp:Real;     // strength of seasonal forcing
  s_peak:Real;    // seasonal peak week
  p_rep:Real;     // proportion of cases reported
  σ2_λ:Real;      // newly infected population variance
  σ2_δ:Real;      // newly incubated population variance
  σ2_γ:Real;      // newly recovered population variance
  σ2_y_δ:Real;    // newly incubated population observation variance

  N:Real;       // total population
  lnS:Real[T];  // susceptible population
  lnE:Real[T];  // incubating population
  lnI:Real[T];  // infectious population
  lnR:Real[T];  // recovered population

  λ:LogNormal[T];  // newly infected population
  δ:LogNormal[T];  // newly incubated population
  γ:LogNormal[T];  // newly recovered population

  y_δ:LogNormal[T];  // observed number of newly infected
  y_N:Integer[T];    // number of serology samples
  y_R:Binomial[T];   // number of positive serology samples

  function simulate() {
    d_inc_h <~ Gaussian(17.8/7.0, pow(2.3/7.0, 2.0));
    d_inf_h <~ Gaussian(4.7/7.0, pow(1.2/7.0, 2.0));
    
    // because there is no truncated Gaussian yet...
    assert d_inc_h > 0.0;
    assert d_inf_h > 0.0;
    
    p_immune <~ Gamma(1.0, 0.06);
    p_risk <~ Uniform(0.1, 1.0);
    R0 <~ Uniform(0.0, 25.0);
    s_amp <~ Uniform(0.0, 1.0);
    s_peak <~ Gaussian(20.0, 4.0);
    p_rep <~ Beta(1.0, 1.0);
    σ2_λ <~ Uniform(0.1, 2.0);
    σ2_δ <~ Uniform(0.1, 2.0);
    σ2_γ <~ Uniform(0.1, 2.0);
    σ2_y_δ <~ Uniform(0.1, 2.0);

    N <- 100000.0;
    lnR[1] <- log(N*p_immune*p_risk);
    lnI[1] <- log(simulate_gamma(1.0, 10.0));
    lnE[1] <- log(simulate_gamma(1.0, 10.0));
    lnS[1] <- log(N - exp(lnI[1]) - exp(lnR[1]) - exp(lnE[1]));
    
    for (t:Integer in 2..T) {
      r_λ:Real <- 1.0/d_inc_h;
      r_δ:Real <- (1.0/d_inf_h)*(1.0 + s_amp*cos(2.0*π*(t - s_peak)/52.0));
      r_γ:Real <- R0/d_inf_h;

      λ[t] ~ LogNormal(log(r_λ) + lnS[t-1] + lnI[t-1] - log(N), σ2_λ);
      δ[t] ~ LogNormal(log(r_δ) + lnE[t-1], σ2_δ);
      γ[t] ~ LogNormal(log(r_γ) + lnI[t-1], σ2_γ);

      y_δ[t] ~ LogNormal(log(p_rep*δ[t]), σ2_y_δ);

      lnS[t] <- lnS[t-1] - λ[t]/exp(lnS[t-1]);
      lnE[t] <- lnE[t-1] + (λ[t] - δ[t])/exp(lnE[t-1]);
      lnI[t] <- lnI[t-1] + (δ[t] - γ[t])/exp(lnI[t-1]);
      lnR[t] <- lnR[t-1] + γ[t]/exp(lnR[t-1]);

      y_R[t] ~ Binomial(y_N[t], exp(lnR[t])/N);
    }
  }

  function input() {
    // read in files
  }

  function output() {
    t:Integer;
    for (t in 1..T) {
      stdout.print(exp(lnS[t]) + ",");
      stdout.print(exp(lnE[t]) + ",");
      stdout.print(exp(lnI[t]) + ",");
      stdout.print(exp(lnR[t]) + ",");
      stdout.print(λ[t] + ",");
      stdout.print(δ[t] + ",");
      stdout.print(γ[t] + "\n");
    }
  }
}
