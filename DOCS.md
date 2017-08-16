
# Global

| Variable | Description |
| --- | --- |
| *inf:[Real64](#real64-0)* | $\infty$ |
| *π:[Real64](#real64-0)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-505) | Create. |
| [Beta](#beta-685) | Create. |
| [Boolean](#boolean-752) | Convert String to Boolean. |
| [Gamma](#gamma-695) | Create. |
| [Gaussian](#gaussian-539) | Create. |
| [Gaussian](#gaussian-577) | Create. |
| [Integer](#integer-766) | Convert Reak to Integer. |
| [Integer](#integer-768) | Convert String to Integer. |
| [Integer32](#integer32-760) | Convert String to Integer32. |
| [Integer64](#integer64-758) | Convert String to Integer64. |
| [Real](#real-762) | Convert String to Real. |
| [Real](#real-764) | Convert Integer to Real. |
| [Real32](#real32-756) | Convert String to Real32. |
| [Real64](#real64-754) | Convert String to Real64. |
| [String](#string-770) | Convert String to String (identity function). |
| [Uniform](#uniform-523) | Create. |
| [adjacent_difference](#adjacent-difference-75) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-129) | Number of columns of a matrix. |
| [columns](#columns-131) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-657) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-71) | Inclusive prefix sum. |
| [identity](#identity-146) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-67) | Inclusive prefix sum. |
| [inverse](#inverse-661) | Inverse of a matrix. |
| [length](#length-77) | Length of a vector. |
| [length](#length-79) | Length of a vector. |
| [llt](#llt-664) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-63) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-140) | Create matrix filled with a given scalar. |
| [max](#max-54) | Maximum of a vector. |
| [min](#min-58) | Minimum of a vector. |
| [norm](#norm-653) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [print](#print-477) | Print scalar. |
| [print](#print-479) | Print scalar. |
| [print](#print-481) | Print scalar. |
| [print](#print-483) | Print scalar. |
| [print](#print-486) | Print vector. |
| [print](#print-489) | Print vector. |
| [print](#print-493) | Print matrix. |
| [print](#print-497) | Print matrix. |
| [printf](#printf-466) | Print with format. |
| [printf](#printf-469) | Print with format. |
| [printf](#printf-472) | Print with format. |
| [printf](#printf-475) | Print with format. |
| [read](#read-44) | Read numbers from a file. |
| [rows](#rows-125) | Number of rows of a matrix. |
| [rows](#rows-127) | Number of rows of a matrix. |
| [scalar](#scalar-133) | Convert single-element matrix to scalar. |
| [scalar](#scalar-81) | Convert single-element vector to scalar. |
| [seed](#seed-46) | Seed the pseudorandom number generator. |
| [solve](#solve-668) | Solve a system of equations. |
| [solve](#solve-672) | Solve a system of equations. |
| [squaredNorm](#squarednorm-655) | Squared norm of a vector. |
| [sum](#sum-50) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-659) | Transpose of a matrix. |
| [vector](#vector-86) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-513) | Build the project. |
| [check](#check-463) | Check the file structure of the project for possible issues. |
| [clean](#clean-462) | Clean the project directory of all build files. |
| [dist](#dist-460) | Build a distributable archive for the project. |
| [docs](#docs-742) | Build the reference documentation for the project. |
| [init](#init-148) | Initialise the working directory for a new project. |
| [install](#install-123) | Install the project. |
| [uninstall](#uninstall-461) | Uninstall the project. |


## Function Details

#### Bernoulli(ρ:[Real](#real-0)) -> [Bernoulli](#bernoulli-0)

<a name="bernoulli-505"></a>

Create.

#### Beta(α:[Real](#real-0), β:[Real](#real-0)) -> [Beta](#beta-0)

<a name="beta-685"></a>

Create.

#### Boolean(s:[String](#string-0)) -> [Boolean](#boolean-0)

<a name="boolean-752"></a>

Convert String to Boolean.

#### Gamma(k:[Real](#real-0), θ:[Real](#real-0)) -> [Gamma](#gamma-0)

<a name="gamma-695"></a>

Create.

#### Gaussian(μ:[Real](#real-0), σ2:[Real](#real-0)) -> [Gaussian](#gaussian-0)

<a name="gaussian-539"></a>

Create.

#### Gaussian(μ:[Real](#real-0)\[\_\], Σ:[Real](#real-0)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-0)

<a name="gaussian-577"></a>

Create.

#### Integer(x:[Real](#real-0)) -> [Integer](#integer-0)

<a name="integer-766"></a>

Convert Reak to Integer.

#### Integer(s:[String](#string-0)) -> [Integer](#integer-0)

<a name="integer-768"></a>

Convert String to Integer.

#### Integer32(s:[String](#string-0)) -> [Integer32](#integer32-0)

<a name="integer32-760"></a>

Convert String to Integer32.

#### Integer64(s:[String](#string-0)) -> [Integer64](#integer64-0)

<a name="integer64-758"></a>

Convert String to Integer64.

#### Real(s:[String](#string-0)) -> [Real](#real-0)

<a name="real-762"></a>

Convert String to Real.

#### Real(x:[Integer](#integer-0)) -> [Real](#real-0)

<a name="real-764"></a>

Convert Integer to Real.

#### Real32(s:[String](#string-0)) -> [Real32](#real32-0)

<a name="real32-756"></a>

Convert String to Real32.

#### Real64(s:[String](#string-0)) -> [Real64](#real64-0)

<a name="real64-754"></a>

Convert String to Real64.

#### String(s:[String](#string-0)) -> [String](#string-0)

<a name="string-770"></a>

Convert String to String (identity function).

#### Uniform(l:[Real](#real-0), u:[Real](#real-0)) -> [Uniform](#uniform-0)

<a name="uniform-523"></a>

Create.

#### adjacent_difference(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="adjacent-difference-75"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="columns-129"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="columns-131"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="determinant-657"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="exclusive-prefix-sum-71"></a>

Inclusive prefix sum.

#### identity(rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="identity-146"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="inclusive-prefix-sum-67"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="inverse-661"></a>

Inverse of a matrix.

#### length(x:[Real](#real-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-77"></a>

Length of a vector.

#### length(x:[Integer](#integer-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-79"></a>

Length of a vector.

#### llt(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="llt-664"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="log-sum-exp-63"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-0), rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="matrix-140"></a>

Create matrix filled with a given scalar.

#### max(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="max-54"></a>

Maximum of a vector.

#### min(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="min-58"></a>

Minimum of a vector.

#### norm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="norm-653"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### print(value:[Boolean](#boolean-0))

<a name="print-477"></a>

Print scalar.

#### print(value:[Integer](#integer-0))

<a name="print-479"></a>

Print scalar.

#### print(value:[Real](#real-0))

<a name="print-481"></a>

Print scalar.

#### print(value:[String](#string-0))

<a name="print-483"></a>

Print scalar.

#### print(x:[Integer](#integer-0)\[\_\])

<a name="print-486"></a>

Print vector.

#### print(x:[Real](#real-0)\[\_\])

<a name="print-489"></a>

Print vector.

#### print(X:[Integer](#integer-0)\[\_,\_\])

<a name="print-493"></a>

Print matrix.

#### print(X:[Real](#real-0)\[\_,\_\])

<a name="print-497"></a>

Print matrix.

#### printf(fmt:[String](#string-0), value:[Boolean](#boolean-0))

<a name="printf-466"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[Integer](#integer-0))

<a name="printf-469"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[Real](#real-0))

<a name="printf-472"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[String](#string-0))

<a name="printf-475"></a>

Print with format. See system `printf`.

#### read(file:[String](#string-0), N:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="read-44"></a>

Read numbers from a file.

#### rows(X:[Real](#real-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-125"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-127"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="scalar-133"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="scalar-81"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-0))

<a name="seed-46"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-0)\[\_,\_\], y:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="solve-668"></a>

Solve a system of equations.

#### solve(X:[Real](#real-0)\[\_,\_\], Y:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="solve-672"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="squarednorm-655"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="sum-50"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="transpose-659"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-0), length:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="vector-86"></a>

Create vector filled with a given scalar.


## Program Details

#### build(include_dir:[String](#string-0), lib_dir:[String](#string-0), share_dir:[String](#string-0), prefix:[String](#string-0), warnings:[Boolean](#boolean-0) <- true, debug:[Boolean](#boolean-0) <- true, verbose:[Boolean](#boolean-0) <- true)

<a name="build-513"></a>

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

<a name="check-463"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-462"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-460"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-742"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-0) <- "untitled")

<a name="init-148"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-123"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-461"></a>

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
| [initialize](#initialize-702) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-0), u:[Gaussian](#gaussian-0), c:[Real](#real-0))

<a name="initialize-702"></a>

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
| [initialize](#initialize-779) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-0)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-0), c:[Real](#real-0)\[\_\])

<a name="initialize-779"></a>

Initialize.


## Bernoulli

<a name="bernoulli-0"></a>

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-0)* | Probability of a true result. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-499) | Simulate. |
| [observe](#observe-501) | Observe. |


### Member Function Details

#### observe(x:[Boolean](#boolean-0)) -> [Real](#real-0)

<a name="observe-501"></a>

Observe.

#### simulate() -> [Boolean](#boolean-0)

<a name="simulate-499"></a>

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
| [simulate](#simulate-677) | Simulate. |
| [observe](#observe-680) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-680"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-677"></a>

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
| [isRoot](#isroot-95) | Is this a root node? |
| [isTerminal](#isterminal-96) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-97) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-98) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-99) | Is this node in the marginalized state? |
| [isRealized](#isrealized-100) | Is this node in the realized state? |
| [isMissing](#ismissing-101) | Is the value of this node missing? |
| [initialize](#initialize-102) | Initialize as a root node. |
| [initialize](#initialize-104) | Initialize as a non-root node. |
| [marginalize](#marginalize-105) | Marginalize the variate. |
| [forward](#forward-106) | Forward simulate the variate. |
| [realize](#realize-107) | Realize the variate. |
| [graft](#graft-108) | Graft the stem to this node. |
| [graft](#graft-110) | Graft the stem to this node. |
| [prune](#prune-111) | Prune the stem from below this node. |
| [setParent](#setparent-113) | Set the parent. |
| [removeParent](#removeparent-114) | Remove the parent. |
| [setChild](#setchild-116) | Set the child. |
| [removeChild](#removechild-117) | Remove the child. |


### Member Function Details

#### forward()

<a name="forward-106"></a>

Forward simulate the variate.

#### graft()

<a name="graft-108"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-0))

<a name="graft-110"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-102"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-0))

<a name="initialize-104"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-0)

<a name="isinitialized-98"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-0)

<a name="ismarginalized-99"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-0)

<a name="ismissing-101"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-0)

<a name="isrealized-100"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-0)

<a name="isroot-95"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-0)

<a name="isterminal-96"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-0)

<a name="isuninitialized-97"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-105"></a>

Marginalize the variate.

#### prune()

<a name="prune-111"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-107"></a>

Realize the variate.

#### removeChild()

<a name="removechild-117"></a>

Remove the child.

#### removeParent()

<a name="removeparent-114"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-0))

<a name="setchild-116"></a>

Set the child.

#### setParent(u:[Delay](#delay-0))

<a name="setparent-113"></a>

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
| [simulate](#simulate-688) | Simulate. |
| [observe](#observe-690) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-690"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-688"></a>

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
| [simulate](#simulate-516) | Simulate. |
| [observe](#observe-518) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-518"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-516"></a>

Simulate.

