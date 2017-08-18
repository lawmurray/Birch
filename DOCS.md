
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-522)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-692)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-132)* | Standard error. |
| *stdout:[StdOutStream](#stdoutstream-121)* | Standard output. |
| *π:[Real64](#real64-692)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-395) | Create. |
| [Beta](#beta-109) | Create. |
| [Boolean](#boolean-886) | Convert other basic types to Boolean. |
| [Gamma](#gamma-387) | Create. |
| [Gaussian](#gaussian-366) | Create. |
| [Gaussian](#gaussian-542) | Create. |
| [Integer](#integer-681) | Convert other basic types to Integer. |
| [Integer32](#integer32-816) | Convert other basic types to Integer32. |
| [Integer64](#integer64-398) | Convert other basic types to Integer64. |
| [Real](#real-875) | Convert other basic types to Real. |
| [Real32](#real32-44) | Convert other basic types to Real32. |
| [Real64](#real64-694) | Convert other basic types to Real64. |
| [String](#string-899) | Convert other basic types to String. |
| [Uniform](#uniform-144) | Create. |
| [abs](#abs-860) | Absolute value. |
| [abs](#abs-442) | Absolute value. |
| [abs](#abs-88) | Absolute value. |
| [abs](#abs-738) | Absolute value. |
| [adjacent_difference](#adjacent-difference-173) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-216) | Number of columns of a matrix. |
| [columns](#columns-218) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-663) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-169) | Inclusive prefix sum. |
| [fclose](#fclose-374) | Close a file. |
| [fopen](#fopen-369) | Open a file for reading. |
| [fopen](#fopen-372) | Open a file. |
| [identity](#identity-233) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-165) | Inclusive prefix sum. |
| [inverse](#inverse-667) | Inverse of a matrix. |
| [isnan](#isnan-96) | Does this have the value NaN? |
| [isnan](#isnan-746) | Does this have the value NaN? |
| [length](#length-962) | Length of a string. |
| [length](#length-111) | Length of a vector. |
| [length](#length-113) | Length of a vector. |
| [llt](#llt-670) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-161) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-227) | Create matrix filled with a given scalar. |
| [max](#max-863) | Maximum of two values. |
| [max](#max-445) | Maximum of two values. |
| [max](#max-91) | Maximum of two values. |
| [max](#max-741) | Maximum of two values. |
| [max](#max-152) | Maximum of a vector. |
| [min](#min-866) | Minimum of two values. |
| [min](#min-448) | Minimum of two values. |
| [min](#min-94) | Minimum of two values. |
| [min](#min-744) | Minimum of two values. |
| [min](#min-156) | Minimum of a vector. |
| [norm](#norm-659) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [read](#read-872) | Read numbers from a file. |
| [rows](#rows-212) | Number of rows of a matrix. |
| [rows](#rows-214) | Number of rows of a matrix. |
| [scalar](#scalar-220) | Convert single-element matrix to scalar. |
| [scalar](#scalar-115) | Convert single-element vector to scalar. |
| [seed](#seed-41) | Seed the pseudorandom number generator. |
| [solve](#solve-674) | Solve a system of equations. |
| [solve](#solve-678) | Solve a system of equations. |
| [squaredNorm](#squarednorm-661) | Squared norm of a vector. |
| [sum](#sum-148) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-665) | Transpose of a matrix. |
| [vector](#vector-120) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-131) | Build the project. |
| [check](#check-350) | Check the file structure of the project for possible issues. |
| [clean](#clean-561) | Clean the project directory of all build files. |
| [dist](#dist-39) | Build a distributable archive for the project. |
| [docs](#docs-235) | Build the reference documentation for the project. |
| [init](#init-691) | Initialise the working directory for a new project. |
| [install](#install-234) | Install the project. |
| [uninstall](#uninstall-210) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-884) | A Boolean value. |
| [File](#file-367) | A file handle. |
| [Integer32](#integer32-814) | A 32-bit integer. |
| [Integer64](#integer64-396) | A 64-bit integer. |
| [Real32](#real32-42) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-692) | A 64-bit (double precision) floating point value. |
| [String](#string-897) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-679) | An integer value of default type. |
| [Real](#real-873) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [AffineGaussian](#affinegaussian-762) | Gaussian that has a mean which is an affine transformation of another Gaussian. |
| [AffineGaussianExpression](#affinegaussianexpression-569) | Expression used to accumulate affine transformations of Gaussians. |
| [AffineMultivariateGaussian](#affinemultivariategaussian-805) | Multivariate Gaussian that has a mean which is an affine transformation of another multivariate Gaussian. |
| [AffineMultivariateGaussianExpression](#affinemultivariategaussianexpression-458) | Expression used to accumulate affine transformations of multivariate Gaussians. |
| [Bernoulli](#bernoulli-392) | Bernoulli distribution. |
| [Beta](#beta-105) | Beta distribution. |
| [Delay](#delay-1001) | Node interface for delayed sampling. |
| [DelayDiagnostics](#delaydiagnostics-522) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayReal](#delayreal-560) | Abstract delay variate with real value. |
| [DelayRealVector](#delayrealvector-787) | Abstract delay variate with real vector value. |
| [FileOutputStream](#fileoutputstream-377) | File output stream. |
| [Gamma](#gamma-383) | Gamma distribution. |
| [Gaussian](#gaussian-362) | Gaussian distribution. |
| [MultivariateGaussian](#multivariategaussian-537) | Multivariate Gaussian distribution. |
| [OutputStream](#outputstream-209) | Output stream. |
| [StdErrStream](#stderrstream-132) | Output stream for stderr. |
| [StdOutStream](#stdoutstream-121) | Output stream for stdout. |
| [Uniform](#uniform-140) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-873)) -> [Bernoulli](#bernoulli-392)

<a name="bernoulli-395"></a>

Create.

#### Beta(α:[Real](#real-873), β:[Real](#real-873)) -> [Beta](#beta-105)

<a name="beta-109"></a>

Create.

#### Boolean(x:[Boolean](#boolean-884)) -> [Boolean](#boolean-884)

<a name="boolean-886"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-873), θ:[Real](#real-873)) -> [Gamma](#gamma-383)

<a name="gamma-387"></a>

Create.

#### Gaussian(μ:[Real](#real-873), σ2:[Real](#real-873)) -> [Gaussian](#gaussian-362)

<a name="gaussian-366"></a>

Create.

#### Gaussian(μ:[Real](#real-873)\[\_\], Σ:[Real](#real-873)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-537)

<a name="gaussian-542"></a>

Create.

#### Integer(x:[Integer64](#integer64-396)) -> [Integer](#integer-679)

<a name="integer-681"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-814)) -> [Integer32](#integer32-814)

<a name="integer32-816"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-396)) -> [Integer64](#integer64-396)

<a name="integer64-398"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-692)) -> [Real](#real-873)

<a name="real-875"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-42)) -> [Real32](#real32-42)

<a name="real32-44"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-692)) -> [Real64](#real64-692)

<a name="real64-694"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-897)) -> [String](#string-897)

<a name="string-899"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-873), u:[Real](#real-873)) -> [Uniform](#uniform-140)

<a name="uniform-144"></a>

Create.

#### abs(x:[Integer32](#integer32-814)) -> [Integer32](#integer32-814)

<a name="abs-860"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-396)) -> [Integer64](#integer64-396)

<a name="abs-442"></a>

Absolute value.

#### abs(x:[Real32](#real32-42)) -> [Real32](#real32-42)

<a name="abs-88"></a>

Absolute value.

#### abs(x:[Real64](#real64-692)) -> [Real64](#real64-692)

<a name="abs-738"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)\[\_\]

<a name="adjacent-difference-173"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-873)\[\_\]) -> [Integer](#integer-679)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-873)\[\_\]) -> [Integer](#integer-679)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-873)\[\_,\_\]) -> [Integer64](#integer64-396)

<a name="columns-216"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-679)\[\_,\_\]) -> [Integer64](#integer64-396)

<a name="columns-218"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-679)\[\_\]) -> [Integer](#integer-679)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-873)\[\_\]) -> [Real](#real-873)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)

<a name="determinant-663"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)\[\_\]

<a name="exclusive-prefix-sum-169"></a>

Inclusive prefix sum.

#### fclose(file:[File](#file-367))

<a name="fclose-374"></a>

Close a file.

#### fopen(file:[String](#string-897)) -> [File](#file-367)

<a name="fopen-369"></a>

Open a file for reading.

  - file : The file name.

#### fopen(file:[String](#string-897), mode:[String](#string-897)) -> [File](#file-367)

<a name="fopen-372"></a>

Open a file.

  - file : The file name.
  - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
    other modes as in system `fopen`.

#### identity(rows:[Integer](#integer-679), columns:[Integer](#integer-679)) -> [Real](#real-873)\[\_,\_\]

<a name="identity-233"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)\[\_\]

<a name="inclusive-prefix-sum-165"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)\[\_,\_\]

<a name="inverse-667"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-42)) -> [Boolean](#boolean-884)

<a name="isnan-96"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-692)) -> [Boolean](#boolean-884)

<a name="isnan-746"></a>

Does this have the value NaN?

#### length(x:[String](#string-897)) -> [Integer](#integer-679)

<a name="length-962"></a>

Length of a string.

#### length(x:[Real](#real-873)\[\_\]) -> [Integer64](#integer64-396)

<a name="length-111"></a>

Length of a vector.

#### length(x:[Integer](#integer-679)\[\_\]) -> [Integer64](#integer64-396)

<a name="length-113"></a>

Length of a vector.

#### llt(X:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)\[\_,\_\]

<a name="llt-670"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="log-sum-exp-161"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-873), rows:[Integer](#integer-679), columns:[Integer](#integer-679)) -> [Real](#real-873)\[\_,\_\]

<a name="matrix-227"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-814), y:[Integer32](#integer32-814)) -> [Integer32](#integer32-814)

<a name="max-863"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-396), y:[Integer64](#integer64-396)) -> [Integer64](#integer64-396)

<a name="max-445"></a>

Maximum of two values.

#### max(x:[Real32](#real32-42), y:[Real32](#real32-42)) -> [Real32](#real32-42)

<a name="max-91"></a>

Maximum of two values.

#### max(x:[Real64](#real64-692), y:[Real64](#real64-692)) -> [Real64](#real64-692)

<a name="max-741"></a>

Maximum of two values.

#### max(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="max-152"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-814), y:[Integer32](#integer32-814)) -> [Integer32](#integer32-814)

<a name="min-866"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-396), y:[Integer64](#integer64-396)) -> [Integer64](#integer64-396)

<a name="min-448"></a>

Minimum of two values.

#### min(x:[Real32](#real32-42), y:[Real32](#real32-42)) -> [Real32](#real32-42)

<a name="min-94"></a>

Minimum of two values.

#### min(x:[Real64](#real64-692), y:[Real64](#real64-692)) -> [Real64](#real64-692)

<a name="min-744"></a>

Minimum of two values.

#### min(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="min-156"></a>

Minimum of a vector.

#### norm(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="norm-659"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-679)\[\_\]) -> [Integer](#integer-679)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### read(file:[String](#string-897), N:[Integer](#integer-679)) -> [Real](#real-873)\[\_\]

<a name="read-872"></a>

Read numbers from a file.

#### rows(X:[Real](#real-873)\[\_,\_\]) -> [Integer64](#integer64-396)

<a name="rows-212"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-679)\[\_,\_\]) -> [Integer64](#integer64-396)

<a name="rows-214"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)

<a name="scalar-220"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="scalar-115"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-679))

<a name="seed-41"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-873)\[\_,\_\], y:[Real](#real-873)\[\_\]) -> [Real](#real-873)\[\_\]

<a name="solve-674"></a>

Solve a system of equations.

#### solve(X:[Real](#real-873)\[\_,\_\], Y:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)\[\_,\_\]

<a name="solve-678"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="squarednorm-661"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-873)\[\_\]) -> [Real](#real-873)

<a name="sum-148"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-873)\[\_\]) -> [Integer](#integer-679)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-873)\[\_,\_\]) -> [Real](#real-873)\[\_,\_\]

<a name="transpose-665"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-873), length:[Integer](#integer-679)) -> [Real](#real-873)\[\_\]

<a name="vector-120"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-897), lib_dir:[String](#string-897), share_dir:[String](#string-897), prefix:[String](#string-897), warnings:[Boolean](#boolean-884) <- true, debug:[Boolean](#boolean-884) <- true, verbose:[Boolean](#boolean-884) <- true)

<a name="build-131"></a>

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

<a name="check-350"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-561"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-39"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-235"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-897) <- "untitled")

<a name="init-691"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-234"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-210"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Basic Type Details

#### type Boolean

<a name="boolean-884"></a>

A Boolean value.

#### type File

<a name="file-367"></a>

A file handle.

#### type Integer32

<a name="integer32-814"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-396"></a>

A 64-bit integer.

#### type Real32

<a name="real32-42"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-692"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-897"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-396)

<a name="integer-679"></a>

An integer value of default type.

#### type Real = [Real64](#real64-692)

<a name="real-873"></a>

A floating point value of default type.


# Class Type Details


## AffineGaussian

<a name="affinegaussian-762"></a>

  * Inherits from *[Gaussian](#gaussian-362)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-873)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-362)* | Mean. |
| *c:[Real](#real-873)* | Additive scalar of affine transformation. |
| *q:[Real](#real-873)* | Variance. |
| *y:[Real](#real-873)* | Marginalized prior mean. |
| *s:[Real](#real-873)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-569"></a>

Expression used to accumulate affine transformations of Gaussians.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-873)* | Multiplicative scalar of affine transformation. |
| *u:[Gaussian](#gaussian-362)* | Parent. |
| *c:[Real](#real-873)* | Additive scalar of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-568) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-873), u:[Gaussian](#gaussian-362), c:[Real](#real-873))

<a name="initialize-568"></a>

Initialize.


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-805"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-537)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-873)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-537)* | Mean. |
| *c:[Real](#real-873)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-873)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-873)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-873)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-458"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-873)\[\_,\_\]* | Matrix of affine transformation. |
| *u:[MultivariateGaussian](#multivariategaussian-537)* | Parent. |
| *c:[Real](#real-873)\[\_\]* | Vector of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-457) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-873)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-537), c:[Real](#real-873)\[\_\])

<a name="initialize-457"></a>

Initialize.


## Bernoulli

<a name="bernoulli-392"></a>

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-873)* | Probability of a true result. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-389) | Simulate. |
| [observe](#observe-391) | Observe. |


### Member Function Details

#### observe(x:[Boolean](#boolean-884)) -> [Real](#real-873)

<a name="observe-391"></a>

Observe.

#### simulate() -> [Boolean](#boolean-884)

<a name="simulate-389"></a>

Simulate.


## Beta

<a name="beta-105"></a>

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-873)* | First shape parameter. |
| *β:[Real](#real-873)* | Second shape parameter. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-101) | Simulate. |
| [observe](#observe-104) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-873)) -> [Real](#real-873)

<a name="observe-104"></a>

Observe.

#### simulate() -> [Real](#real-873)

<a name="simulate-101"></a>

Simulate.


## Delay

<a name="delay-1001"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-679)* | State of the variate. |
| *missing:[Boolean](#boolean-884)* | Is the value missing? |
| *parent:[Delay](#delay-1001)?* | Parent. |
| *child:[Delay](#delay-1001)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-679)* | Unique id for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-972) | Is this a root node? |
| [isTerminal](#isterminal-973) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-974) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-975) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-976) | Is this node in the marginalized state? |
| [isRealized](#isrealized-977) | Is this node in the realized state? |
| [isMissing](#ismissing-978) | Is the value of this node missing? |
| [initialize](#initialize-979) | Initialize as a root node. |
| [initialize](#initialize-981) | Initialize as a non-root node. |
| [marginalize](#marginalize-982) | Marginalize the variate. |
| [forward](#forward-983) | Forward simulate the variate. |
| [realize](#realize-984) | Realize the variate. |
| [graft](#graft-985) | Graft the stem to this node. |
| [graft](#graft-987) | Graft the stem to this node. |
| [prune](#prune-988) | Prune the stem from below this node. |
| [setParent](#setparent-990) | Set the parent. |
| [removeParent](#removeparent-991) | Remove the parent. |
| [setChild](#setchild-993) | Set the child. |
| [removeChild](#removechild-994) | Remove the child. |
| [register](#register-999) | Register with the diagnostic handler. |
| [trigger](#trigger-1000) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### forward()

<a name="forward-983"></a>

Forward simulate the variate.

#### graft()

<a name="graft-985"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-1001))

<a name="graft-987"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-979"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-1001))

<a name="initialize-981"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-884)

<a name="isinitialized-975"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-884)

<a name="ismarginalized-976"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-884)

<a name="ismissing-978"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-884)

<a name="isrealized-977"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-884)

<a name="isroot-972"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-884)

<a name="isterminal-973"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-884)

<a name="isuninitialized-974"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-982"></a>

Marginalize the variate.

#### prune()

<a name="prune-988"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-984"></a>

Realize the variate.

#### register()

<a name="register-999"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-994"></a>

Remove the child.

#### removeParent()

<a name="removeparent-991"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-1001))

<a name="setchild-993"></a>

Set the child.

#### setParent(u:[Delay](#delay-1001))

<a name="setparent-990"></a>

Set the parent.

#### trigger()

<a name="trigger-1000"></a>

Trigger an event with the diagnostic handler.


## DelayDiagnostics

<a name="delaydiagnostics-522"></a>

Outputs graphical representations of the delayed sampling state for
diagnostic purposes.

  - N : Maximum number of nodes

To use, before running any code that uses delayed sampling, construct a
`DelayDiagnostic` object with sufficient capacity to hold all nodes.
Then, use `name()` to give a name to each node that will be of interest
to output, and optionally `position()` to give an explicit position.
Variates that are not named are not output.

See the Birch example programs, e.g. `delay_triplet` and `delay_kalman`
for an example of how this is done.

On each event, will output a `*.dot` file into `diagnostics/stateN.dot`,
where `N` is the event number. If positions have been explicitly given
using `position()`, it is recommended that these are compiled with
`neato`, otherwise, if positions have not been explicitly given so that
automatic layout is desired, use `dot`, e.g.

    dot -Tpdf diagnostics/state1.dot > diagnostics/state1.pdf

| Member Variable | Description |
| --- | --- |
| *nodes:[Delay](#delay-1001)?\[\_\]* | Registered nodes. |
| *names:[String](#string-897)?\[\_\]* | Names of the nodes. |
| *xs:[Integer](#integer-679)?\[\_\]* | $x$-coordinates of the nodes. |
| *ys:[Integer](#integer-679)?\[\_\]* | $y$-coordinates of the nodes. |
| *n:[Integer](#integer-679)* | Number of nodes that have been registered. |
| *nevents:[Integer](#integer-679)* | Number of events that have been triggered. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-508) | Register a new node. |
| [name](#name-511) | Set the name of a node. |
| [position](#position-515) | Set the position of a node. |
| [trigger](#trigger-516) | Trigger an event. |
| [dot](#dot-521) | Output a dot graph of the current state. |


### Member Function Details

#### dot()

<a name="dot-521"></a>

Output a dot graph of the current state.

#### name(id:[Integer](#integer-679), name:[String](#string-897))

<a name="name-511"></a>

Set the name of a node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-679), x:[Integer](#integer-679), y:[Integer](#integer-679))

<a name="position-515"></a>

Set the position of a node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

#### register(o:[Delay](#delay-1001)) -> [Integer](#integer-679)

<a name="register-508"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-516"></a>

Trigger an event.


## DelayReal

<a name="delayreal-560"></a>

  * Inherits from *[Delay](#delay-1001)*

Abstract delay variate with real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-873)* | Value assignment. |
| *[String](#string-897)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-873)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-873)* | Value. |
| *w:[Real](#real-873)* | Weight. |


## DelayRealVector

<a name="delayrealvector-787"></a>

  * Inherits from *[Delay](#delay-1001)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-873)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-873)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-873)\[\_\]* | Value. |
| *w:[Real](#real-873)* | Weight. |


## FileOutputStream

<a name="fileoutputstream-377"></a>

  * Inherits from *[OutputStream](#outputstream-209)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-376) | Close the file. |


### Member Function Details

#### close()

<a name="close-376"></a>

Close the file.


## Gamma

<a name="gamma-383"></a>

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-873)* | Shape. |
| *θ:[Real](#real-873)* | Scale. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-380) | Simulate. |
| [observe](#observe-382) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-873)) -> [Real](#real-873)

<a name="observe-382"></a>

Observe.

#### simulate() -> [Real](#real-873)

<a name="simulate-380"></a>

Simulate.


## Gaussian

<a name="gaussian-362"></a>

  * Inherits from *[DelayReal](#delayreal-560)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-873)* | Mean. |
| *σ2:[Real](#real-873)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-537"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-787)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-873)\[\_\]* | Mean. |
| *Σ:[Real](#real-873)\[\_,\_\]* | Covariance matrix. |


## OutputStream

<a name="outputstream-209"></a>

Output stream.

| Member Function | Brief description |
| --- | --- |
| [printf](#printf-177) | Print with format. |
| [printf](#printf-180) | Print with format. |
| [printf](#printf-183) | Print with format. |
| [printf](#printf-186) | Print with format. |
| [print](#print-188) | Print scalar. |
| [print](#print-190) | Print scalar. |
| [print](#print-192) | Print scalar. |
| [print](#print-194) | Print scalar. |
| [print](#print-197) | Print vector. |
| [print](#print-200) | Print vector. |
| [print](#print-204) | Print matrix. |
| [print](#print-208) | Print matrix. |


### Member Function Details

#### print(value:[Boolean](#boolean-884))

<a name="print-188"></a>

Print scalar.

#### print(value:[Integer](#integer-679))

<a name="print-190"></a>

Print scalar.

#### print(value:[Real](#real-873))

<a name="print-192"></a>

Print scalar.

#### print(value:[String](#string-897))

<a name="print-194"></a>

Print scalar.

#### print(x:[Integer](#integer-679)\[\_\])

<a name="print-197"></a>

Print vector.

#### print(x:[Real](#real-873)\[\_\])

<a name="print-200"></a>

Print vector.

#### print(X:[Integer](#integer-679)\[\_,\_\])

<a name="print-204"></a>

Print matrix.

#### print(X:[Real](#real-873)\[\_,\_\])

<a name="print-208"></a>

Print matrix.

#### printf(fmt:[String](#string-897), value:[Boolean](#boolean-884))

<a name="printf-177"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-897), value:[Integer](#integer-679))

<a name="printf-180"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-897), value:[Real](#real-873))

<a name="printf-183"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-897), value:[String](#string-897))

<a name="printf-186"></a>

Print with format. See system `printf`.


## StdErrStream

<a name="stderrstream-132"></a>

  * Inherits from *[OutputStream](#outputstream-209)*

Output stream for stderr.


## StdOutStream

<a name="stdoutstream-121"></a>

  * Inherits from *[OutputStream](#outputstream-209)*

Output stream for stdout.


## Uniform

<a name="uniform-140"></a>

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-873)* | Lower bound. |
| *u:[Real](#real-873)* | Upper bound. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-137) | Simulate. |
| [observe](#observe-139) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-873)) -> [Real](#real-873)

<a name="observe-139"></a>

Observe.

#### simulate() -> [Real](#real-873)

<a name="simulate-137"></a>

Simulate.

