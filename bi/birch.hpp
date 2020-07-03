/**
 * @file
 */
#pragma once

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
#include <filesystem>
namespace fs = std::filesystem;
#else
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
namespace fs = boost::filesystem;
#endif

#undef line // line() macro in indentable_iostream clashes with property_tree
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/trim.hpp"

#include <getopt.h>
#include <dlfcn.h>
#ifdef HAVE_LIBEXPLAIN_SYSTEM_H
#include <libexplain/system.h>
#endif
