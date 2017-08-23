
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-526)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-767)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-105)* | Standard error. |
| *stdout:[StdOutStream](#stdoutstream-94)* | Standard output. |
| *π:[Real64](#real64-767)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-393) | Create. |
| [Beta](#beta-606) | Create a Beta distribution. |
| [Boolean](#boolean-961) | Convert other basic types to Boolean. |
| [Gamma](#gamma-589) | Create Gamma distribution. |
| [Gaussian](#gaussian-383) | Create a Gaussian distribution. |
| [Gaussian](#gaussian-546) | Create. |
| [Integer](#integer-756) | Convert other basic types to Integer. |
| [Integer32](#integer32-891) | Convert other basic types to Integer32. |
| [Integer64](#integer64-396) | Convert other basic types to Integer64. |
| [Real](#real-950) | Convert other basic types to Real. |
| [Real32](#real32-41) | Convert other basic types to Real32. |
| [Real64](#real64-769) | Convert other basic types to Real64. |
| [String](#string-974) | Convert other basic types to String. |
| [Uniform](#uniform-367) | Create a Uniform distribution. |
| [abs](#abs-935) | Absolute value. |
| [abs](#abs-440) | Absolute value. |
| [abs](#abs-85) | Absolute value. |
| [abs](#abs-813) | Absolute value. |
| [adjacent_difference](#adjacent-difference-353) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-168) | Number of columns of a matrix. |
| [columns](#columns-170) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-738) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-349) | Inclusive prefix sum. |
| [fclose](#fclose-572) | Close a file. |
| [fopen](#fopen-567) | Open a file for reading. |
| [fopen](#fopen-570) | Open a file. |
| [identity](#identity-185) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-345) | Inclusive prefix sum. |
| [inverse](#inverse-742) | Inverse of a matrix. |
| [isnan](#isnan-93) | Does this have the value NaN? |
| [isnan](#isnan-821) | Does this have the value NaN? |
| [length](#length-1037) | Length of a string. |
| [length](#length-608) | Length of a vector. |
| [length](#length-610) | Length of a vector. |
| [llt](#llt-745) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-341) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-179) | Create matrix filled with a given scalar. |
| [max](#max-938) | Maximum of two values. |
| [max](#max-443) | Maximum of two values. |
| [max](#max-88) | Maximum of two values. |
| [max](#max-816) | Maximum of two values. |
| [max](#max-332) | Maximum of a vector. |
| [min](#min-941) | Minimum of two values. |
| [min](#min-446) | Minimum of two values. |
| [min](#min-91) | Minimum of two values. |
| [min](#min-819) | Minimum of two values. |
| [min](#min-336) | Minimum of a vector. |
| [norm](#norm-734) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [random_bernoulli](#random-bernoulli-313) | Simulate a Bernoulli variate. |
| [random_gamma](#random-gamma-322) | Simulate a Gamma variate. |
| [random_gaussian](#random-gaussian-319) | Simulate a Gaussian variate. |
| [random_uniform](#random-uniform-316) | Simulate a Uniform variate. |
| [read](#read-947) | Read numbers from a file. |
| [rows](#rows-164) | Number of rows of a matrix. |
| [rows](#rows-166) | Number of rows of a matrix. |
| [scalar](#scalar-172) | Convert single-element matrix to scalar. |
| [scalar](#scalar-612) | Convert single-element vector to scalar. |
| [seed](#seed-311) | Seed the pseudorandom number generator. |
| [solve](#solve-749) | Solve a system of equations. |
| [solve](#solve-753) | Solve a system of equations. |
| [squaredNorm](#squarednorm-736) | Squared norm of a vector. |
| [sum](#sum-328) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-740) | Transpose of a matrix. |
| [vector](#vector-617) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-104) | Build the project. |
| [check](#check-324) | Check the file structure of the project for possible issues. |
| [clean](#clean-636) | Clean the project directory of all build files. |
| [dist](#dist-323) | Build a distributable archive for the project. |
| [docs](#docs-187) | Build the reference documentation for the project. |
| [init](#init-766) | Initialise the working directory for a new project. |
| [install](#install-186) | Install the project. |
| [uninstall](#uninstall-162) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-959) | A Boolean value. |
| [File](#file-565) | A file handle. |
| [Integer32](#integer32-889) | A 32-bit integer. |
| [Integer64](#integer64-394) | A 64-bit integer. |
| [Real32](#real32-39) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-767) | A 64-bit (double precision) floating point value. |
| [String](#string-972) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-754) | An integer value of default type. |
| [Real](#real-948) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [AffineGaussian](#affinegaussian-837) | Gaussian that has a mean which is an affine transformation of another Gaussian. |
| [AffineGaussianExpression](#affinegaussianexpression-644) | Expression used to accumulate affine transformations of Gaussians. |
| [AffineMultivariateGaussian](#affinemultivariategaussian-880) | Multivariate Gaussian that has a mean which is an affine transformation of another multivariate Gaussian. |
| [AffineMultivariateGaussianExpression](#affinemultivariategaussianexpression-456) | Expression used to accumulate affine transformations of multivariate Gaussians. |
| [Bernoulli](#bernoulli-390) | Bernoulli distribution. |
| [Beta](#beta-602) | Beta distribution. |
| [Delay](#delay-1079) | Node interface for delayed sampling. |
| [DelayBoolean](#delayboolean-161) | Abstract delay variate with Boolean value. |
| [DelayDiagnostics](#delaydiagnostics-526) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayInteger](#delayinteger-635) | Abstract delay variate with Integer value. |
| [DelayReal](#delayreal-564) | Abstract delay variate with Real value. |
| [DelayRealVector](#delayrealvector-862) | Abstract delay variate with real vector value. |
| [FileOutputStream](#fileoutputstream-575) | File output stream. |
| [Gamma](#gamma-585) | Gamma distribution. |
| [Gaussian](#gaussian-379) | Gaussian distribution. |
| [MultivariateGaussian](#multivariategaussian-541) | Multivariate Gaussian distribution. |
| [OutputStream](#outputstream-143) | Output stream. |
| [StdErrStream](#stderrstream-105) | Output stream for stderr. |
| [StdOutStream](#stdoutstream-94) | Output stream for stdout. |
| [Uniform](#uniform-363) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-948)) -> [Bernoulli](#bernoulli-390)

<a name="bernoulli-393"></a>

Create.

#### Beta(α:[Real](#real-948), β:[Real](#real-948)) -> [Beta](#beta-602)

<a name="beta-606"></a>

Create a Beta distribution.

#### Boolean(x:[Boolean](#boolean-959)) -> [Boolean](#boolean-959)

<a name="boolean-961"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-948), θ:[Real](#real-948)) -> [Gamma](#gamma-585)

<a name="gamma-589"></a>

Create Gamma distribution.

#### Gaussian(μ:[Real](#real-948), σ2:[Real](#real-948)) -> [Gaussian](#gaussian-379)

<a name="gaussian-383"></a>

Create a Gaussian distribution.

#### Gaussian(μ:[Real](#real-948)\[\_\], Σ:[Real](#real-948)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-541)

<a name="gaussian-546"></a>

Create.

#### Integer(x:[Integer64](#integer64-394)) -> [Integer](#integer-754)

<a name="integer-756"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-889)) -> [Integer32](#integer32-889)

<a name="integer32-891"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-394)) -> [Integer64](#integer64-394)

<a name="integer64-396"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-767)) -> [Real](#real-948)

<a name="real-950"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="real32-41"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-767)) -> [Real64](#real64-767)

<a name="real64-769"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-972)) -> [String](#string-972)

<a name="string-974"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-948), u:[Real](#real-948)) -> [Uniform](#uniform-363)

<a name="uniform-367"></a>

Create a Uniform distribution.

#### abs(x:[Integer32](#integer32-889)) -> [Integer32](#integer32-889)

<a name="abs-935"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-394)) -> [Integer64](#integer64-394)

<a name="abs-440"></a>

Absolute value.

#### abs(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="abs-85"></a>

Absolute value.

#### abs(x:[Real64](#real64-767)) -> [Real64](#real64-767)

<a name="abs-813"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)\[\_\]

<a name="adjacent-difference-353"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-948)\[\_\]) -> [Integer](#integer-754)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-948)\[\_\]) -> [Integer](#integer-754)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-948)\[\_,\_\]) -> [Integer64](#integer64-394)

<a name="columns-168"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-754)\[\_,\_\]) -> [Integer64](#integer64-394)

<a name="columns-170"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-754)\[\_\]) -> [Integer](#integer-754)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-948)\[\_\]) -> [Real](#real-948)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)

<a name="determinant-738"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)\[\_\]

<a name="exclusive-prefix-sum-349"></a>

Inclusive prefix sum.

#### fclose(file:[File](#file-565))

<a name="fclose-572"></a>

Close a file.

#### fopen(file:[String](#string-972)) -> [File](#file-565)

<a name="fopen-567"></a>

Open a file for reading.

  - file : The file name.

#### fopen(file:[String](#string-972), mode:[String](#string-972)) -> [File](#file-565)

<a name="fopen-570"></a>

Open a file.

  - file : The file name.
  - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
    other modes as in system `fopen`.

#### identity(rows:[Integer](#integer-754), columns:[Integer](#integer-754)) -> [Real](#real-948)\[\_,\_\]

<a name="identity-185"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)\[\_\]

<a name="inclusive-prefix-sum-345"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)\[\_,\_\]

<a name="inverse-742"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-39)) -> [Boolean](#boolean-959)

<a name="isnan-93"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-767)) -> [Boolean](#boolean-959)

<a name="isnan-821"></a>

Does this have the value NaN?

#### length(x:[String](#string-972)) -> [Integer](#integer-754)

<a name="length-1037"></a>

Length of a string.

#### length(x:[Real](#real-948)\[\_\]) -> [Integer64](#integer64-394)

<a name="length-608"></a>

Length of a vector.

#### length(x:[Integer](#integer-754)\[\_\]) -> [Integer64](#integer64-394)

<a name="length-610"></a>

Length of a vector.

#### llt(X:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)\[\_,\_\]

<a name="llt-745"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="log-sum-exp-341"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-948), rows:[Integer](#integer-754), columns:[Integer](#integer-754)) -> [Real](#real-948)\[\_,\_\]

<a name="matrix-179"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-889), y:[Integer32](#integer32-889)) -> [Integer32](#integer32-889)

<a name="max-938"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-394), y:[Integer64](#integer64-394)) -> [Integer64](#integer64-394)

<a name="max-443"></a>

Maximum of two values.

#### max(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="max-88"></a>

Maximum of two values.

#### max(x:[Real64](#real64-767), y:[Real64](#real64-767)) -> [Real64](#real64-767)

<a name="max-816"></a>

Maximum of two values.

#### max(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="max-332"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-889), y:[Integer32](#integer32-889)) -> [Integer32](#integer32-889)

<a name="min-941"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-394), y:[Integer64](#integer64-394)) -> [Integer64](#integer64-394)

<a name="min-446"></a>

Minimum of two values.

#### min(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="min-91"></a>

Minimum of two values.

#### min(x:[Real64](#real64-767), y:[Real64](#real64-767)) -> [Real64](#real64-767)

<a name="min-819"></a>

Minimum of two values.

#### min(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="min-336"></a>

Minimum of a vector.

#### norm(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="norm-734"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-754)\[\_\]) -> [Integer](#integer-754)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### random_bernoulli(ρ:[Real](#real-948)) -> [Boolean](#boolean-959)

<a name="random-bernoulli-313"></a>

Simulate a Bernoulli variate.

#### random_gamma(k:[Real](#real-948), θ:[Real](#real-948)) -> [Real](#real-948)

<a name="random-gamma-322"></a>

Simulate a Gamma variate.

#### random_gaussian(μ:[Real](#real-948), σ2:[Real](#real-948)) -> [Real](#real-948)

<a name="random-gaussian-319"></a>

Simulate a Gaussian variate.

#### random_uniform(l:[Real](#real-948), u:[Real](#real-948)) -> [Real](#real-948)

<a name="random-uniform-316"></a>

Simulate a Uniform variate.

#### read(file:[String](#string-972), N:[Integer](#integer-754)) -> [Real](#real-948)\[\_\]

<a name="read-947"></a>

Read numbers from a file.

#### rows(X:[Real](#real-948)\[\_,\_\]) -> [Integer64](#integer64-394)

<a name="rows-164"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-754)\[\_,\_\]) -> [Integer64](#integer64-394)

<a name="rows-166"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)

<a name="scalar-172"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="scalar-612"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-754))

<a name="seed-311"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-948)\[\_,\_\], y:[Real](#real-948)\[\_\]) -> [Real](#real-948)\[\_\]

<a name="solve-749"></a>

Solve a system of equations.

#### solve(X:[Real](#real-948)\[\_,\_\], Y:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)\[\_,\_\]

<a name="solve-753"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="squarednorm-736"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-948)\[\_\]) -> [Real](#real-948)

<a name="sum-328"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-948)\[\_\]) -> [Integer](#integer-754)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-948)\[\_,\_\]) -> [Real](#real-948)\[\_,\_\]

<a name="transpose-740"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-948), length:[Integer](#integer-754)) -> [Real](#real-948)\[\_\]

<a name="vector-617"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-972), lib_dir:[String](#string-972), share_dir:[String](#string-972), prefix:[String](#string-972), warnings:[Boolean](#boolean-959) <- true, debug:[Boolean](#boolean-959) <- true, verbose:[Boolean](#boolean-959) <- true)

<a name="build-104"></a>

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

<a name="check-324"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-636"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-323"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-187"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-972) <- "untitled")

<a name="init-766"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-186"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-162"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Basic Type Details

#### type Boolean

<a name="boolean-959"></a>

A Boolean value.

#### type File

<a name="file-565"></a>

A file handle.

#### type Integer32

<a name="integer32-889"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-394"></a>

A 64-bit integer.

#### type Real32

<a name="real32-39"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-767"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-972"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-394)

<a name="integer-754"></a>

An integer value of default type.

#### type Real = [Real64](#real64-767)

<a name="real-948"></a>

A floating point value of default type.


# Class Type Details


## AffineGaussian

<a name="affinegaussian-837"></a>

  * Inherits from *[Gaussian](#gaussian-379)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-948)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-379)* | Mean. |
| *c:[Real](#real-948)* | Additive scalar of affine transformation. |
| *q:[Real](#real-948)* | Variance. |
| *y:[Real](#real-948)* | Marginalized prior mean. |
| *s:[Real](#real-948)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-644"></a>

Expression used to accumulate affine transformations of Gaussians.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-948)* | Multiplicative scalar of affine transformation. |
| *u:[Gaussian](#gaussian-379)* | Parent. |
| *c:[Real](#real-948)* | Additive scalar of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-643) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-948), u:[Gaussian](#gaussian-379), c:[Real](#real-948))

<a name="initialize-643"></a>

Initialize.


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-880"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-541)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-948)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-541)* | Mean. |
| *c:[Real](#real-948)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-948)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-948)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-948)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-456"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-948)\[\_,\_\]* | Matrix of affine transformation. |
| *u:[MultivariateGaussian](#multivariategaussian-541)* | Parent. |
| *c:[Real](#real-948)\[\_\]* | Vector of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-455) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-948)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-541), c:[Real](#real-948)\[\_\])

<a name="initialize-455"></a>

Initialize.


## Bernoulli

<a name="bernoulli-390"></a>

  * Inherits from *[DelayBoolean](#delayboolean-161)*

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-948)* | Probability of a true result. |


## Beta

<a name="beta-602"></a>

  * Inherits from *[DelayReal](#delayreal-564)*

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-948)* | First shape parameter. |
| *β:[Real](#real-948)* | Second shape parameter. |


## Delay

<a name="delay-1079"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-754)* | State of the variate. |
| *missing:[Boolean](#boolean-959)* | Is the value missing? |
| *parent:[Delay](#delay-1079)?* | Parent. |
| *child:[Delay](#delay-1079)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-754)* | Unique id for delayed sampling diagnostics. |
| *nforward:[Integer](#integer-754)* | Number of observations absorbed in forward pass, for delayed sampling diagnostics. |
| *nbackward:[Integer](#integer-754)* | Number of observations absorbed in backward pass, for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-1049) | Is this a root node? |
| [isTerminal](#isterminal-1050) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-1051) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-1052) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-1053) | Is this node in the marginalized state? |
| [isRealized](#isrealized-1054) | Is this node in the realized state? |
| [isMissing](#ismissing-1055) | Is the value of this node missing? |
| [initialize](#initialize-1056) | Initialize as a root node. |
| [initialize](#initialize-1058) | Initialize as a non-root node. |
| [update](#update-1059) | Update the variate. |
| [marginalize](#marginalize-1060) | Marginalize the variate. |
| [forward](#forward-1061) | Forward simulate the variate. |
| [realize](#realize-1062) | Realize the variate. |
| [graft](#graft-1063) | Graft the stem to this node. |
| [graft](#graft-1065) | Graft the stem to this node. |
| [prune](#prune-1066) | Prune the stem from below this node. |
| [setParent](#setparent-1068) | Set the parent. |
| [removeParent](#removeparent-1069) | Remove the parent. |
| [setChild](#setchild-1071) | Set the child. |
| [removeChild](#removechild-1072) | Remove the child. |
| [register](#register-1077) | Register with the diagnostic handler. |
| [trigger](#trigger-1078) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### forward()

<a name="forward-1061"></a>

Forward simulate the variate.

#### graft()

<a name="graft-1063"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-1079))

<a name="graft-1065"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-1056"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-1079))

<a name="initialize-1058"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-959)

<a name="isinitialized-1052"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-959)

<a name="ismarginalized-1053"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-959)

<a name="ismissing-1055"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-959)

<a name="isrealized-1054"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-959)

<a name="isroot-1049"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-959)

<a name="isterminal-1050"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-959)

<a name="isuninitialized-1051"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-1060"></a>

Marginalize the variate.

#### prune()

<a name="prune-1066"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-1062"></a>

Realize the variate.

#### register()

<a name="register-1077"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-1072"></a>

Remove the child.

#### removeParent()

<a name="removeparent-1069"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-1079))

<a name="setchild-1071"></a>

Set the child.

#### setParent(u:[Delay](#delay-1079))

<a name="setparent-1068"></a>

Set the parent.

#### trigger()

<a name="trigger-1078"></a>

Trigger an event with the diagnostic handler.

#### update()

<a name="update-1059"></a>

Update the variate.


## DelayBoolean

<a name="delayboolean-161"></a>

  * Inherits from *[Delay](#delay-1079)*

Abstract delay variate with Boolean value.

| Assignment | Description |
| --- | --- |
| *[Boolean](#boolean-959)* | Value assignment. |
| *[String](#string-972)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Boolean](#boolean-959)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Boolean](#boolean-959)* | Value. |
| *w:[Real](#real-948)* | Weight. |


## DelayDiagnostics

<a name="delaydiagnostics-526"></a>

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
| *nodes:[Delay](#delay-1079)?\[\_\]* | Registered nodes. |
| *names:[String](#string-972)?\[\_\]* | Names of the nodes. |
| *xs:[Integer](#integer-754)?\[\_\]* | $x$-coordinates of the nodes. |
| *ys:[Integer](#integer-754)?\[\_\]* | $y$-coordinates of the nodes. |
| *n:[Integer](#integer-754)* | Number of nodes that have been registered. |
| *noutputs:[Integer](#integer-754)* | Number of graphs that have been output. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-506) | Register a new node. |
| [name](#name-509) | Set the name of a node. |
| [position](#position-513) | Set the position of a node. |
| [trigger](#trigger-516) | Trigger an event. |
| [dot](#dot-525) | Output a dot graph of the current state. |


### Member Function Details

#### dot()

<a name="dot-525"></a>

Output a dot graph of the current state.

#### name(id:[Integer](#integer-754), name:[String](#string-972))

<a name="name-509"></a>

Set the name of a node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-754), x:[Integer](#integer-754), y:[Integer](#integer-754))

<a name="position-513"></a>

Set the position of a node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

#### register(o:[Delay](#delay-1079)) -> [Integer](#integer-754)

<a name="register-506"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-516"></a>

Trigger an event.


## DelayInteger

<a name="delayinteger-635"></a>

  * Inherits from *[Delay](#delay-1079)*

Abstract delay variate with Integer value.

| Assignment | Description |
| --- | --- |
| *[Integer](#integer-754)* | Value assignment. |
| *[String](#string-972)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Integer](#integer-754)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Integer](#integer-754)* | Value. |
| *w:[Real](#real-948)* | Weight. |


## DelayReal

<a name="delayreal-564"></a>

  * Inherits from *[Delay](#delay-1079)*

Abstract delay variate with Real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-948)* | Value assignment. |
| *[String](#string-972)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-948)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-948)* | Value. |
| *w:[Real](#real-948)* | Weight. |


## DelayRealVector

<a name="delayrealvector-862"></a>

  * Inherits from *[Delay](#delay-1079)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-948)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-948)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-948)\[\_\]* | Value. |
| *w:[Real](#real-948)* | Weight. |


## FileOutputStream

<a name="fileoutputstream-575"></a>

  * Inherits from *[OutputStream](#outputstream-143)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-574) | Close the file. |


### Member Function Details

#### close()

<a name="close-574"></a>

Close the file.


## Gamma

<a name="gamma-585"></a>

  * Inherits from *[DelayReal](#delayreal-564)*

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-948)* | Shape. |
| *θ:[Real](#real-948)* | Scale. |


## Gaussian

<a name="gaussian-379"></a>

  * Inherits from *[DelayReal](#delayreal-564)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-948)* | Mean. |
| *σ2:[Real](#real-948)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-541"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-862)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-948)\[\_\]* | Mean. |
| *Σ:[Real](#real-948)\[\_,\_\]* | Covariance matrix. |


## OutputStream

<a name="outputstream-143"></a>

Output stream.

| Member Function | Brief description |
| --- | --- |
| [printf](#printf-111) | Print with format. |
| [printf](#printf-114) | Print with format. |
| [printf](#printf-117) | Print with format. |
| [printf](#printf-120) | Print with format. |
| [print](#print-122) | Print scalar. |
| [print](#print-124) | Print scalar. |
| [print](#print-126) | Print scalar. |
| [print](#print-128) | Print scalar. |
| [print](#print-131) | Print vector. |
| [print](#print-134) | Print vector. |
| [print](#print-138) | Print matrix. |
| [print](#print-142) | Print matrix. |


### Member Function Details

#### print(value:[Boolean](#boolean-959))

<a name="print-122"></a>

Print scalar.

#### print(value:[Integer](#integer-754))

<a name="print-124"></a>

Print scalar.

#### print(value:[Real](#real-948))

<a name="print-126"></a>

Print scalar.

#### print(value:[String](#string-972))

<a name="print-128"></a>

Print scalar.

#### print(x:[Integer](#integer-754)\[\_\])

<a name="print-131"></a>

Print vector.

#### print(x:[Real](#real-948)\[\_\])

<a name="print-134"></a>

Print vector.

#### print(X:[Integer](#integer-754)\[\_,\_\])

<a name="print-138"></a>

Print matrix.

#### print(X:[Real](#real-948)\[\_,\_\])

<a name="print-142"></a>

Print matrix.

#### printf(fmt:[String](#string-972), value:[Boolean](#boolean-959))

<a name="printf-111"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-972), value:[Integer](#integer-754))

<a name="printf-114"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-972), value:[Real](#real-948))

<a name="printf-117"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-972), value:[String](#string-972))

<a name="printf-120"></a>

Print with format. See system `printf`.


## StdErrStream

<a name="stderrstream-105"></a>

  * Inherits from *[OutputStream](#outputstream-143)*

Output stream for stderr.


## StdOutStream

<a name="stdoutstream-94"></a>

  * Inherits from *[OutputStream](#outputstream-143)*

Output stream for stdout.


## Uniform

<a name="uniform-363"></a>

  * Inherits from *[DelayReal](#delayreal-564)*

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-948)* | Lower bound. |
| *u:[Real](#real-948)* | Upper bound. |

