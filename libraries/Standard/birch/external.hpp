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
