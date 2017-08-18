
# Global

| Variable | Description |
| --- | --- |
| *inf:[Real64](#real64-0)* | $\infty$ |
| *π:[Real64](#real64-0)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-132) | Create. |
| [Beta](#beta-588) | Create. |
| [Boolean](#boolean-602) | Convert other basic types to Boolean. |
| [Gamma](#gamma-267) | Create. |
| [Gaussian](#gaussian-211) | Create. |
| [Gaussian](#gaussian-504) | Create. |
| [Integer](#integer-591) | Convert other basic types to Integer. |
| [Integer32](#integer32-270) | Convert other basic types to Integer32. |
| [Integer64](#integer64-839) | Convert other basic types to Integer64. |
| [Real](#real-696) | Convert other basic types to Real. |
| [Real32](#real32-143) | Convert other basic types to Real32. |
| [Real64](#real64-641) | Convert other basic types to Real64. |
| [String](#string-401) | Convert other basic types to String. |
| [Uniform](#uniform-925) | Create. |
| [abs](#abs-314) | Absolute value. |
| [abs](#abs-883) | Absolute value. |
| [abs](#abs-187) | Absolute value. |
| [abs](#abs-685) | Absolute value. |
| [adjacent_difference](#adjacent-difference-75) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-344) | Number of columns of a matrix. |
| [columns](#columns-346) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-560) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-71) | Inclusive prefix sum. |
| [identity](#identity-361) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-67) | Inclusive prefix sum. |
| [inverse](#inverse-564) | Inverse of a matrix. |
| [isnan](#isnan-195) | Does this have the value NaN? |
| [isnan](#isnan-693) | Does this have the value NaN? |
| [length](#length-434) | Length of a string. |
| [length](#length-77) | Length of a vector. |
| [length](#length-79) | Length of a vector. |
| [llt](#llt-567) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-63) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-355) | Create matrix filled with a given scalar. |
| [max](#max-317) | Maximum of two values. |
| [max](#max-886) | Maximum of two values. |
| [max](#max-190) | Maximum of two values. |
| [max](#max-688) | Maximum of two values. |
| [max](#max-54) | Maximum of a vector. |
| [min](#min-320) | Minimum of two values. |
| [min](#min-889) | Minimum of two values. |
| [min](#min-193) | Minimum of two values. |
| [min](#min-691) | Minimum of two values. |
| [min](#min-58) | Minimum of a vector. |
| [norm](#norm-556) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [print](#print-104) | Print scalar. |
| [print](#print-106) | Print scalar. |
| [print](#print-108) | Print scalar. |
| [print](#print-110) | Print scalar. |
| [print](#print-113) | Print vector. |
| [print](#print-116) | Print vector. |
| [print](#print-120) | Print matrix. |
| [print](#print-124) | Print matrix. |
| [printf](#printf-93) | Print with format. |
| [printf](#printf-96) | Print with format. |
| [printf](#printf-99) | Print with format. |
| [printf](#printf-102) | Print with format. |
| [read](#read-44) | Read numbers from a file. |
| [rows](#rows-340) | Number of rows of a matrix. |
| [rows](#rows-342) | Number of rows of a matrix. |
| [scalar](#scalar-348) | Convert single-element matrix to scalar. |
| [scalar](#scalar-81) | Convert single-element vector to scalar. |
| [seed](#seed-46) | Seed the pseudorandom number generator. |
| [solve](#solve-571) | Solve a system of equations. |
| [solve](#solve-575) | Solve a system of equations. |
| [squaredNorm](#squarednorm-558) | Squared norm of a vector. |
| [sum](#sum-50) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-562) | Transpose of a matrix. |
| [vector](#vector-86) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-140) | Build the project. |
| [check](#check-90) | Check the file structure of the project for possible issues. |
| [clean](#clean-89) | Clean the project directory of all build files. |
| [dist](#dist-819) | Build a distributable archive for the project. |
| [docs](#docs-88) | Build the reference documentation for the project. |
| [init](#init-638) | Initialise the working directory for a new project. |
| [install](#install-362) | Install the project. |
| [uninstall](#uninstall-87) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-600) | A Boolean value. |
| [Integer32](#integer32-268) | A 32-bit integer. |
| [Integer64](#integer64-837) | A 64-bit integer. |
| [Real32](#real32-141) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-639) | A 64-bit (double precision) floating point value. |
| [String](#string-399) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-589) | An integer value of default type. |
| [Real](#real-694) | A floating point value of default type. |


## Function Details

#### Bernoulli(ρ:[Real](#real-0)) -> [Bernoulli](#bernoulli-0)

<a name="bernoulli-132"></a>

Create.

#### Beta(α:[Real](#real-0), β:[Real](#real-0)) -> [Beta](#beta-0)

<a name="beta-588"></a>

Create.

#### Boolean(x:[Boolean](#boolean-0)) -> [Boolean](#boolean-0)

<a name="boolean-602"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-0), θ:[Real](#real-0)) -> [Gamma](#gamma-0)

<a name="gamma-267"></a>

Create.

#### Gaussian(μ:[Real](#real-0), σ2:[Real](#real-0)) -> [Gaussian](#gaussian-0)

<a name="gaussian-211"></a>

Create.

#### Gaussian(μ:[Real](#real-0)\[\_\], Σ:[Real](#real-0)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-0)

<a name="gaussian-504"></a>

Create.

#### Integer(x:[Integer64](#integer64-0)) -> [Integer](#integer-0)

<a name="integer-591"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-0)) -> [Integer32](#integer32-0)

<a name="integer32-270"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-0)) -> [Integer64](#integer64-0)

<a name="integer64-839"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-0)) -> [Real](#real-0)

<a name="real-696"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-0)) -> [Real32](#real32-0)

<a name="real32-143"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-0)) -> [Real64](#real64-0)

<a name="real64-641"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-0)) -> [String](#string-0)

<a name="string-401"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-0), u:[Real](#real-0)) -> [Uniform](#uniform-0)

<a name="uniform-925"></a>

Create.

#### abs(x:[Integer32](#integer32-0)) -> [Integer32](#integer32-0)

<a name="abs-314"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-0)) -> [Integer64](#integer64-0)

<a name="abs-883"></a>

Absolute value.

#### abs(x:[Real32](#real32-0)) -> [Real32](#real32-0)

<a name="abs-187"></a>

Absolute value.

#### abs(x:[Real64](#real64-0)) -> [Real64](#real64-0)

<a name="abs-685"></a>

Absolute value.

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

<a name="columns-344"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="columns-346"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="determinant-560"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="exclusive-prefix-sum-71"></a>

Inclusive prefix sum.

#### identity(rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="identity-361"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="inclusive-prefix-sum-67"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="inverse-564"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-0)) -> [Boolean](#boolean-0)

<a name="isnan-195"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-0)) -> [Boolean](#boolean-0)

<a name="isnan-693"></a>

Does this have the value NaN?

#### length(x:[String](#string-0)) -> [Integer](#integer-0)

<a name="length-434"></a>

Length of a string.

#### length(x:[Real](#real-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-77"></a>

Length of a vector.

#### length(x:[Integer](#integer-0)\[\_\]) -> [Integer64](#integer64-0)

<a name="length-79"></a>

Length of a vector.

#### llt(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="llt-567"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="log-sum-exp-63"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-0), rows:[Integer](#integer-0), columns:[Integer](#integer-0)) -> [Real](#real-0)\[\_,\_\]

<a name="matrix-355"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-0), y:[Integer32](#integer32-0)) -> [Integer32](#integer32-0)

<a name="max-317"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-0), y:[Integer64](#integer64-0)) -> [Integer64](#integer64-0)

<a name="max-886"></a>

Maximum of two values.

#### max(x:[Real32](#real32-0), y:[Real32](#real32-0)) -> [Real32](#real32-0)

<a name="max-190"></a>

Maximum of two values.

#### max(x:[Real64](#real64-0), y:[Real64](#real64-0)) -> [Real64](#real64-0)

<a name="max-688"></a>

Maximum of two values.

#### max(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="max-54"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-0), y:[Integer32](#integer32-0)) -> [Integer32](#integer32-0)

<a name="min-320"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-0), y:[Integer64](#integer64-0)) -> [Integer64](#integer64-0)

<a name="min-889"></a>

Minimum of two values.

#### min(x:[Real32](#real32-0), y:[Real32](#real32-0)) -> [Real32](#real32-0)

<a name="min-193"></a>

Minimum of two values.

#### min(x:[Real64](#real64-0), y:[Real64](#real64-0)) -> [Real64](#real64-0)

<a name="min-691"></a>

Minimum of two values.

#### min(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="min-58"></a>

Minimum of a vector.

#### norm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="norm-556"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### print(value:[Boolean](#boolean-0))

<a name="print-104"></a>

Print scalar.

#### print(value:[Integer](#integer-0))

<a name="print-106"></a>

Print scalar.

#### print(value:[Real](#real-0))

<a name="print-108"></a>

Print scalar.

#### print(value:[String](#string-0))

<a name="print-110"></a>

Print scalar.

#### print(x:[Integer](#integer-0)\[\_\])

<a name="print-113"></a>

Print vector.

#### print(x:[Real](#real-0)\[\_\])

<a name="print-116"></a>

Print vector.

#### print(X:[Integer](#integer-0)\[\_,\_\])

<a name="print-120"></a>

Print matrix.

#### print(X:[Real](#real-0)\[\_,\_\])

<a name="print-124"></a>

Print matrix.

#### printf(fmt:[String](#string-0), value:[Boolean](#boolean-0))

<a name="printf-93"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[Integer](#integer-0))

<a name="printf-96"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[Real](#real-0))

<a name="printf-99"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-0), value:[String](#string-0))

<a name="printf-102"></a>

Print with format. See system `printf`.

#### read(file:[String](#string-0), N:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="read-44"></a>

Read numbers from a file.

#### rows(X:[Real](#real-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-340"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-0)\[\_,\_\]) -> [Integer64](#integer64-0)

<a name="rows-342"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)

<a name="scalar-348"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="scalar-81"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-0))

<a name="seed-46"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-0)\[\_,\_\], y:[Real](#real-0)\[\_\]) -> [Real](#real-0)\[\_\]

<a name="solve-571"></a>

Solve a system of equations.

#### solve(X:[Real](#real-0)\[\_,\_\], Y:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="solve-575"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="squarednorm-558"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-0)\[\_\]) -> [Real](#real-0)

<a name="sum-50"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-0)\[\_\]) -> [Integer](#integer-0)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-0)\[\_,\_\]) -> [Real](#real-0)\[\_,\_\]

<a name="transpose-562"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-0), length:[Integer](#integer-0)) -> [Real](#real-0)\[\_\]

<a name="vector-86"></a>

Create vector filled with a given scalar.


## Program Details

#### build(include_dir:[String](#string-0), lib_dir:[String](#string-0), share_dir:[String](#string-0), prefix:[String](#string-0), warnings:[Boolean](#boolean-0) <- true, debug:[Boolean](#boolean-0) <- true, verbose:[Boolean](#boolean-0) <- true)

<a name="build-140"></a>

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

<a name="check-90"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-89"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-819"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-88"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-0) <- "untitled")

<a name="init-638"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-362"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-87"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


## Basic Type Details

#### type Boolean

<a name="boolean-600"></a>

A Boolean value.

#### type Integer32

<a name="integer32-268"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-837"></a>

A 64-bit integer.

#### type Real32

<a name="real32-141"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-639"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-399"></a>

A string value.


## Alias Type Details

#### type Integer = [Integer64](#integer64-0)

<a name="integer-589"></a>

An integer value of default type.

#### type Real = [Real64](#real64-0)

<a name="real-694"></a>

A floating point value of default type.


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
| [initialize](#initialize-218) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-0), u:[Gaussian](#gaussian-0), c:[Real](#real-0))

<a name="initialize-218"></a>

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
| [initialize](#initialize-443) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-0)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-0), c:[Real](#real-0)\[\_\])

<a name="initialize-443"></a>

Initialize.


## Bernoulli

<a name="bernoulli-0"></a>

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-0)* | Probability of a true result. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-126) | Simulate. |
| [observe](#observe-128) | Observe. |


### Member Function Details

#### observe(x:[Boolean](#boolean-0)) -> [Real](#real-0)

<a name="observe-128"></a>

Observe.

#### simulate() -> [Boolean](#boolean-0)

<a name="simulate-126"></a>

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
| [simulate](#simulate-580) | Simulate. |
| [observe](#observe-583) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-583"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-580"></a>

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
| [isRoot](#isroot-371) | Is this a root node? |
| [isTerminal](#isterminal-372) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-373) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-374) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-375) | Is this node in the marginalized state? |
| [isRealized](#isrealized-376) | Is this node in the realized state? |
| [isMissing](#ismissing-377) | Is the value of this node missing? |
| [initialize](#initialize-378) | Initialize as a root node. |
| [initialize](#initialize-380) | Initialize as a non-root node. |
| [marginalize](#marginalize-381) | Marginalize the variate. |
| [forward](#forward-382) | Forward simulate the variate. |
| [realize](#realize-383) | Realize the variate. |
| [graft](#graft-384) | Graft the stem to this node. |
| [graft](#graft-386) | Graft the stem to this node. |
| [prune](#prune-387) | Prune the stem from below this node. |
| [setParent](#setparent-389) | Set the parent. |
| [removeParent](#removeparent-390) | Remove the parent. |
| [setChild](#setchild-392) | Set the child. |
| [removeChild](#removechild-393) | Remove the child. |


### Member Function Details

#### forward()

<a name="forward-382"></a>

Forward simulate the variate.

#### graft()

<a name="graft-384"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-0))

<a name="graft-386"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-378"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-0))

<a name="initialize-380"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-0)

<a name="isinitialized-374"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-0)

<a name="ismarginalized-375"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-0)

<a name="ismissing-377"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-0)

<a name="isrealized-376"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-0)

<a name="isroot-371"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-0)

<a name="isterminal-372"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-0)

<a name="isuninitialized-373"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-381"></a>

Marginalize the variate.

#### prune()

<a name="prune-387"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-383"></a>

Realize the variate.

#### removeChild()

<a name="removechild-393"></a>

Remove the child.

#### removeParent()

<a name="removeparent-390"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-0))

<a name="setchild-392"></a>

Set the child.

#### setParent(u:[Delay](#delay-0))

<a name="setparent-389"></a>

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
| [simulate](#simulate-260) | Simulate. |
| [observe](#observe-262) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-262"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-260"></a>

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
| [simulate](#simulate-918) | Simulate. |
| [observe](#observe-920) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-0)) -> [Real](#real-0)

<a name="observe-920"></a>

Observe.

#### simulate() -> [Real](#real-0)

<a name="simulate-918"></a>

Simulate.

