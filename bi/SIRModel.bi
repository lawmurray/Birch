/**
 * SIR (susceptible-infectious-recovered) model for infectious disease
 * outbreaks in epidemiology.
 *
 * ### Usage
 *
 * Run with:
 *
 *     birch sample \
 *       --model SIRModel \
 *       --input-file input/russian_influenza.json \
 *       --output-file output/russian_influenza.json \
 *       --ncheckpoints 14 \
 *       --nparticles 100 \
 *       --nsamples 10
 *
 * The data set is of an outbreak of Russian influenza at a boy's boarding
 * school in northern England [(Anonymous, 1978)](../#references). The
 * model on which this is based is described in
 * [Murray et. al. (2018)](../#references).
 *
 * Any of the parameters and the initial conditions can be clamped by
 * modifying the input file, `input/russian_influenza.json`.
 *
 * ### Details 
 *
 * The parameter model is given by:
 *   $$\begin{align}
 *   \lambda &\sim \mathrm{Gamma}(2,5) \\
 *   \delta &\sim \mathrm{Beta}(2,2) \\
 *   \gamma &\sim \mathrm{Beta}(2,2),
 *   \end{align}$$
 * where $\lambda$ is a rate of interaction in the population, $\delta$ the
 * probability of infection when a susceptible individual interacts with an
 * infectious individual, and $\gamma$ the daily recovery probability.
 *
 * The initial model for time $t = 0$ is:
 *   $$\begin{align}
 *   s_0 &= 760 \\
 *   i_0 &= 3 \\
 *   r_0 &= 0.
 *   \end{align}$$
 *
 * The transition model for time $t$ is:
 *   $$\begin{align}
 *   \tau_t &\sim \mathrm{Binomial}\left(s_{t-1}, 1 - \exp\left(\frac{-\lambda
 *   i_{t-1} }{s_{t-1} + i_{t-1} + r_{t-1}}\right) \right) \\
 *   \Delta i_t &\sim \mathrm{Binomial}(\tau_t, \delta) \\
 *   \Delta r_t &\sim \mathrm{Binomial}(i_{t-1}, \gamma),
 *   \end{align}$$
 * where $\tau_t$ is the number of interactions between infectious and
 * susceptible individuals, $\Delta i_t$ the number of newly infected
 * individuals, and $\Delta r_t$ the number of newly recovered individuals.
 *
 * Population counts are then updated:
 * $$\begin{align}
 * s_t &= s_{t-1} - \Delta i_t \\
 * i_t &= i_{t-1} + \Delta i_t - \Delta r_t \\
 * r_t &= r_{t-1} + \Delta r_t.
 * \end{align}$$
 */
class SIRModel = MarkovModel<SIRState,SIRParameter>;
