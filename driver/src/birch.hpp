/**
 * @file
 */
#pragma once

#ifdef __FreeBSD__
#define _WITH_GETLINE /* For getline */
#include <sys/wait.h>  /* For WIF.. */
#endif

#include <vector>
#include <stack>
#include <list>
#include <set>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <regex>
#include <thread>
#include <iomanip>
#include <locale>
#include <codecvt>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cassert>

#if __cplusplus > 201402L
/* for C++17 and above, use the STL filesystem library */
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
namespace fs_stream = std;
#define FS_OWNER_EXE fs::perms::owner_exec
#else
/* otherwise use the boost file system library */
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
namespace fs = boost::filesystem;
namespace fs_stream = boost::filesystem;
#define FS_OWNER_EXE fs::perms::owner_exe
#endif

#include <getopt.h>
#include <dlfcn.h>
#include <yaml.h>
#include <glob.h>
