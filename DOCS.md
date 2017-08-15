
# Global

| Variable | Description |
| --- | --- |
| *inf:[Real64](#real64-0)* | $\infty$ |
| *π:[Real64](#real64-0)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-493) | Create. |
| [Beta](#beta-673) | Create. |
| [Boolean](#boolean-740) | Convert String to Boolean. |
| [Gamma](#gamma-683) | Create. |
| [Gaussian](#gaussian-527) | Create. |
| [Gaussian](#gaussian-565) | Create. |
| [Integer](#integer-754) | Convert Reak to Integer. |
| [Integer](#integer-756) | Convert String to Integer. |
| [Integer32](#integer32-748) | Convert String to Integer32. |
| [Integer64](#integer64-746) | Convert String to Integer64. |
| [Real](#real-750) | Convert String to Real. |
| [Real](#real-752) | Convert Integer to Real. |
| [Real32](#real32-744) | Convert String to Real32. |
| [Real64](#real64-742) | Convert String to Real64. |
| [String](#string-758) | Convert String to String (identity function). |
| [Uniform](#uniform-511) | Create. |
| [adjacent_difference](#adjacent-difference-69) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-123) | Number of columns of a matrix. |
| [columns](#columns-125) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-645) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-65) | Inclusive prefix sum. |
| [identity](#identity-140) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-61) | Inclusive prefix sum. |
| [inverse](#inverse-649) | Inverse of a matrix. |
| [length](#length-71) | Length of a vector. |
| [length](#length-73) | Length of a vector. |
| [llt](#llt-652) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-57) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-134) | Create matrix filled with a given scalar. |
| [max](#max-48) | Maximum of a vector. |
| [min](#min-52) | Minimum of a vector. |
| [norm](#norm-641) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [read](#read-485) | Read numbers from a file. |
| [rows](#rows-119) | Number of rows of a matrix. |
| [rows](#rows-121) | Number of rows of a matrix. |
| [scalar](#scalar-127) | Convert single-element matrix to scalar. |
| [scalar](#scalar-75) | Convert single-element vector to scalar. |
| [seed](#seed-40) | Seed the pseudorandom number generator. |
| [solve](#solve-656) | Solve a system of equations. |
| [solve](#solve-660) | Solve a system of equations. |
| [squaredNorm](#squarednorm-643) | Squared norm of a vector. |
| [sum](#sum-44) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-647) | Transpose of a matrix. |
| [vector](#vector-80) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-501) | Build the project. |
| [check](#check-457) | Check the file structure of the project for possible issues. |
| [clean](#clean-456) | Clean the project directory of all build files. |
| [dist](#dist-454) | Build a distributable archive for the project. |
| [docs](#docs-730) | Build the reference documentation for the project. |
| [init](#init-142) | Initialise the working directory for a new project. |
| [install](#install-117) | Install the project. |
| [uninstall](#uninstall-455) | Uninstall the project. |


## Function Details

#### Bernoulli(ρ:[Real](#real-0)) -> [Bernoulli](#bernoulli-0)

<a name="bernoulli-493"></a>

Create.

#### Beta(α:[Real](#real-0), β:[Real](#real-0)) -> [Beta](#beta-0)

<a name="beta-673"></a>

Create.

#### Boolean(s:[String](#string-0)) -> [Boolean](#boolean-0)

<a name="boolean-740"></a>

Convert String to Boolean.

#### Gamma(k:[Real](#real-0), θ:[Real](#real-0)) -> [Gamma](#gamma-0)

<a name="gamma-683"></a>

Create.

#### Gaussian(μ:[Real](#real-0), σ2:[Real](#real-0)) -> [Gaussian](#gaussian-0)

<a name="gaussian-527"></a>

Create.

#### Gaussian(μ:[Real](#real-0)\[\_\], Σ:[Real](#real-0)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-0)

<a name="gaussian-565"></a>

Create.

#### Integer(x:[Real](#real-0)) -> [Integer](#integer-0)

<a name="integer-754"></a>

Convert Reak to Integer.

#### Integer(s:[String](#string-0)) -> [Integer](#integer-0)

<a name="integer-756"></a>

Convert String to Integer.

#### Integer32(s:[String](#string-0)) -> [Integer32](#integer32-0)

<a name="integer32-748"></a>

Convert String to Integer32.

#### Integer64(s:[String](#string-0)) -> [Integer64](#integer64-0)

<a name="integer64-746"></a>

Convert String to Integer64.

#### Real(s:[String](#string-0)) -> [Real](#real-0)

<a name="real-750"></a>

Convert String to Real.

#### Real(x:[Integer](#integer-0)) -> [Real](#real-0)

<a name="real-752"></a>

Convert Integer to Real.

#### Real32(s:[String](#string-0)) -> [Real32](#real32-0)

<a name="real32-744"></a>

Convert String to Real32.

#### Real64(s:[String](#string-0)) -> [Real64](#real64-0)

<a name="real64-742"></a>

Convert String to Real64.

#### String(s:[String](#string-0)) -> [String](#string-0)

<a name="string-758"></a>

Convert String to String (identity function).

#### Uniform(l:[Real](#real-0), u:[Real](#real-0)) -> [Uniform](#uniform-0)

<a name="uniform-511"></a>

Create.

#### adjacent_difference(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="adjacent-difference-69"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="columns-123"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="columns-125"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="determinant-645"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="exclusive-prefix-sum-65"></a>

Inclusive prefix sum.

#### identity(rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="identity-140"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="inclusive-prefix-sum-61"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="inverse-649"></a>

Inverse of a matrix.

#### length(x:[Real](#real-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-71"></a>

Length of a vector.

#### length(x:[Integer](#integer-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-73"></a>

Length of a vector.

#### llt(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="llt-652"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="log-sum-exp-57"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-0), rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="matrix-134"></a>

Create matrix filled with a given scalar.

#### max(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="max-48"></a>

Maximum of a vector.

#### min(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="min-52"></a>

Minimum of a vector.

#### norm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="norm-641"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### read(file:[String](#string-0), N:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="read-485"></a>

Read numbers from a file.

#### rows(X:[Real](#real-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-119"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-121"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="scalar-127"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="scalar-75"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-0))

<a name="seed-40"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-0)\[\_,\_\], y:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="solve-656"></a>

Solve a system of equations.

#### solve(X:[Real](#real-0)\[\_,\_\], Y:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="solve-660"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="squarednorm-643"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="sum-44"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="transpose-647"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-0), length:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="vector-80"></a>

Create vector filled with a given scalar.


## Program Details

#### build(include_dir:[String](#string-0), lib_dir:[String](#string-0), share_dir:[String](#string-0), prefix:[String](#string-0), warnings:[Boolean](#boolean-0) <- true, debug:[Boolean](#boolean-0) <- true, verbose:[Boolean](#boolean-0) <- true)

<a name="build-501"></a>

Build the project.

  - `--include-dir` : Add search directory for header files.
  - `--lib-dir` : Add search directory for library files.
  - `--share-dir` : Add search directory for data files.

These three options are analogous to their counterparts for a C/C++
compiler, and specify the locations in which the Birch compiler should
search for headers (both Birch and C/C++ headers), installed libraries and
installed data files. They may be given multiple times to specify multiple
directories in the order in which they are to be searched.

After searching these directories, the Birch compiler will search the
environment variables `BIRCH_INCLUDE_PATH`, `BIRCH_LIBRARY_PATH` and
`BIRCH_SHARE_PATH`, followed by the directories of the compiler's own
installation, followed by the system-wide locations `/usr/local/` and
`/usr/`.

  - `--prefix` : Installation prefix (default platform-specific).
  - `--enable-std` / `--disable-std` : Enable/disable the standard library.
  - `--enable-warnings` / `--disable-warnings` : Enable/disable warnings.
  - `--enable-debug` / `--disable-debug` : Enable/disable debug mode.
  - `--enable-verbose` / `--disable-verbose` : Verbose mode.

#### check()

<a name="check-457"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-456"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-454"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-730"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-0) <- "untitled")

<a name="init-142"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-117"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-455"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Classes


## AffineGaussian

<a name="affinegaussian-0"></a>

  * Inherits from *[Gaussian](#gaussian-0)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-0)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-0)* | Mean. |
| *c:[Real](#real-0)* | Additive scalar of affine transformation. |
| *q:[Real](#real-0)* | Variance. |
| *y:[Real](#real-0)* | Marginalized prior mean. |
| *s:[Real](#real-0)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-0"></a>

Expression used to accumulate affine transformations of Gaussians.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-0)* | Multiplicative scalar of affine transformation. |
| *u:[Gaussian](#gaussian-0)* | Parent. |
| *c:[Real](#real-0)* | Additive scalar of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-690) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-0), u:[Gaussian](#gaussian-0), c:[Real](#real-0))

<a name="initialize-690"></a>

Initialize.


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-0"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-0)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-0)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-0)* | Mean. |
| *c:[Real](#real-0)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-0)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-0)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-0)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-0"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-0)\[\_,\_\]* | Matrix of affine transformation. |
| *u:[MultivariateGaussian](#multivariategaussian-0)* | Parent. |
| *c:[Real](#real-0)\[\_\]* | Vector of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-767) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-0)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-0), c:[Real](#real-0)\[\_\])

<a name="initialize-767"></a>

Initialize.


## Bernoulli

<a name="bernoulli-0"></a>

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-0)* | Probability of a true result. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-487) | Simulate. |
| [observe](#observe-489) | Observe. |


### Member Function Details

#### observe(x:[Boolean](#boolean-0)) -> [Real](#real-0)

<a name="observe-489"></a>

Observe.

#### simulate() -> [Boolean](#boolean-0)

<a name="simulate-487"></a>

Simulate.


## Beta

<a name="beta-0"></a>

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-0)* | First shape parameter. |
| *β:[Real](#real-0)* | Second shape parameter. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-665) | Simulate. |
| [observe](#observe-668) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-668"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-665"></a>

Simulate.


## Delay

<a name="delay-0"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-0)* | State of the variate. |
| *missing:[Boolean](#boolean-0)* | Is the value missing? |
| *parent:[Delay](#delay-0)?* | Parent. |
| *child:[Delay](#delay-0)?* | Child, if one exists and it is on the stem. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-89) | Is this a root node? |
| [isTerminal](#isterminal-90) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-91) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-92) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-93) | Is this node in the marginalized state? |
| [isRealized](#isrealized-94) | Is this node in the realized state? |
| [isMissing](#ismissing-95) | Is the value of this node missing? |
| [initialize](#initialize-96) | Initialize as a root node. |
| [initialize](#initialize-98) | Initialize as a non-root node. |
| [marginalize](#marginalize-99) | Marginalize the variate. |
| [forward](#forward-100) | Forward simulate the variate. |
| [realize](#realize-101) | Realize the variate. |
| [graft](#graft-102) | Graft the stem to this node. |
| [graft](#graft-104) | Graft the stem to this node. |
| [prune](#prune-105) | Prune the stem from below this node. |
| [setParent](#setparent-107) | Set the parent. |
| [removeParent](#removeparent-108) | Remove the parent. |
| [setChild](#setchild-110) | Set the child. |
| [removeChild](#removechild-111) | Remove the child. |


### Member Function Details

#### forward()

<a name="forward-100"></a>

Forward simulate the variate.

#### graft()

<a name="graft-102"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-0))

<a name="graft-104"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-96"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-0))

<a name="initialize-98"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-0)

<a name="isinitialized-92"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-0)

<a name="ismarginalized-93"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-0)

<a name="ismissing-95"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-0)

<a name="isrealized-94"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-0)

<a name="isroot-89"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-0)

<a name="isterminal-90"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-0)

<a name="isuninitialized-91"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-99"></a>

Marginalize the variate.

#### prune()

<a name="prune-105"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-101"></a>

Realize the variate.

#### removeChild()

<a name="removechild-111"></a>

Remove the child.

#### removeParent()

<a name="removeparent-108"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-0))

<a name="setchild-110"></a>

Set the child.

#### setParent(u:[Delay](#delay-0))

<a name="setparent-107"></a>

Set the parent.


## DelayReal

<a name="delayreal-0"></a>

  * Inherits from *[Delay](#delay-0)*

Abstract delay variate with real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-0)* | Value assignment. |
| *[String](#string-0)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-0)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-0)* | Value. |
| *w:[Real](#real-0)* | Weight. |


## DelayRealVector

<a name="delayrealvector-0"></a>

  * Inherits from *[Delay](#delay-0)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-0)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-0)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-0)\[\_\]* | Value. |
| *w:[Real](#real-0)* | Weight. |


## Gamma

<a name="gamma-0"></a>

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-0)* | Shape. |
| *θ:[Real](#real-0)* | Scale. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-676) | Simulate. |
| [observe](#observe-678) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-678"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-676"></a>

Simulate.


## Gaussian

<a name="gaussian-0"></a>

  * Inherits from *[DelayReal](#delayreal-0)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-0)* | Mean. |
| *σ2:[Real](#real-0)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-0"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-0)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-0)\[\_\]* | Mean. |
| *Σ:[Real](#real-0)\[\_,\_\]* | Covariance matrix. |


## Uniform

<a name="uniform-0"></a>

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-0)* | Lower bound. |
| *u:[Real](#real-0)* | Upper bound. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-504) | Simulate. |
| [observe](#observe-506) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-506"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-504"></a>

Simulate.

