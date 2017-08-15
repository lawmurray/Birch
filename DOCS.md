
# Global

| Program | Brief description |
| --- | --- |
| [delay_iid](#delay-iid-107) | Demonstrates multiple observations in an array, used to estimate a single parameter. |
| [delay_kalman](#delay-kalman-100) | Demonstrates sampling from a univariate linear-Gaussian state-space model. |
| [delay_rbpf](#delay-rbpf-93) | Demonstrates a particle filter over a nonlinear state-space model with linear substructure. |
| [delay_spike_and_slab](#delay-spike-and-slab-49) | Demonstrates how delayed sampling can yield to different runtime states through a stochastic branch, inspired by a spike-and-slab prior. |
| [delay_triplet](#delay-triplet-53) | Demonstrates sampling from a triplet of Gaussian random variables, with zero or more of them given a value on the command line. |
| [demo_echo](#demo-echo-37) | Echo string to terminal. |
| [demo_hello](#demo-hello-17) | Hello world. |
| [demo_lambda](#demo-lambda-94) | Demonstrates the use of a lambda function to delay the evaluation of an expression. |


## Program Details

#### delay_iid(μ:[Real](#real-0) <- 0.0, σ2:[Real](#real-0) <- 1.0, N:[Integer](#integer-0) <- 100)

<a name="delay-iid-107"></a>

Demonstrates multiple observations in an array, used to estimate a
single parameter.

  - μ  : True mean of the observations.
  - σ2 : True variance of the observations.
  - N  : Number of observations.

Outputs a sample from the posterior distribution of the mean, assuming a
`Gaussian(0.0, 1.0)` prior and Gaussian likelihood with known variance.

#### delay_kalman(a:[Real](#real-0) <- 0.9, T:[Integer](#integer-0) <- 10)

<a name="delay-kalman-100"></a>

Demonstrates sampling from a univariate linear-Gaussian state-space model.

  - a : Autoregressive coefficient.
  - T : Number of time steps.

#### delay_rbpf(N:[Integer](#integer-0) <- 100, T:[Integer](#integer-0) <- 10)

<a name="delay-rbpf-93"></a>

Demonstrates a particle filter over a nonlinear state-space model with
linear substructure. With delayed sampling enabled, this automatically
yields a Rao--Blackwellized particle filter with locally-optimal proposal.

  - N : Number of particles.
  - T : Number of time steps.

To disable delayed sampling, change the `~` operators to `<~` in the
`initial` and `transition` functions of the `Example` class.

#### delay_spike_and_slab()

<a name="delay-spike-and-slab-49"></a>

Demonstrates how delayed sampling can yield to different runtime states
through a stochastic branch, inspired by a spike-and-slab prior. Outputs
whether the variable `y` is marginalized or realized at the end of the
program. This is random in each run.

#### delay_triplet(x:[Gaussian](#gaussian-0), y:[Gaussian](#gaussian-0), z:[Gaussian](#gaussian-0))

<a name="delay-triplet-53"></a>

Demonstrates sampling from a triplet of Gaussian random variables, with
zero or more of them given a value on the command line.

  - x : Value of the first variable.
  - y : Value of the second variable.
  - z : Value of the third variable.

#### demo_echo(message:[String](#string-0) <- "")

<a name="demo-echo-37"></a>

Echo string to terminal.

#### demo_hello()

<a name="demo-hello-17"></a>

Hello world.

#### demo_lambda()

<a name="demo-lambda-94"></a>

Demonstrates the use of a lambda function to delay the evaluation of an
expression.

