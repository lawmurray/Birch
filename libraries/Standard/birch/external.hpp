/**
 * @file
 */
#pragma once

#include <unordered_map>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

#include <dlfcn.h>
#include <unistd.h>
#include <yaml.h>
#include <getopt.h>

#ifdef __FreeBSD__
#include <sys/wait.h>  /* for WIF.. */
#endif

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/pareto.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/weibull.hpp>

#if defined(HAVE_FILESYSTEM)
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(HAVE_EXPERIMENTAL_FILESYSTEM)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
