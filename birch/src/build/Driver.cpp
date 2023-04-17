/**
 * @file
 */
#include "src/build/Driver.hpp"

#include "src/build/MetaParser.hpp"
#include "src/build/Compiler.hpp"
#include "src/primitive/system.hpp"
#include "src/generate/MarkdownGenerator.hpp"
#include "src/primitive/string.hpp"
#include "src/exception/DriverException.hpp"

birch::Driver::Driver(int argc, char** argv) :
    packageName("Untitled"),
    packageVersion("0.0.0"),
    unit("dir"),
    precision("double"),
    backend("eigen"),
    jobs(std::thread::hardware_concurrency()),
    enableSingle(true),
    enableDouble(true),
    enableStatic(false),
    enableShared(true),
    enableStandalone(false),
    enableOpenmp(true),
    enableAssert(true),
    enableOptimize(true),
    enableDebug(true),
    enableCoverage(false),
    enableWarnings(true),
    enableNotes(true),
    enableTranslate(true),
    enableVerbose(false),
    newBootstrap(false),
    newConfigure(false),
    newMake(false) {
  /* environment */
  char* BIRCH_UNIT = getenv("BIRCH_UNIT");
  char* BIRCH_PRECISION = getenv("BIRCH_PRECISION");
  char* BIRCH_BACKEND = getenv("BIRCH_BACKEND");
  char* BIRCH_PREFIX = getenv("BIRCH_PREFIX");
  char* BIRCH_SHARE_PATH = getenv("BIRCH_SHARE_PATH");
  char* BIRCH_INCLUDE_PATH = getenv("BIRCH_INCLUDE_PATH");
  char* BIRCH_LIBRARY_PATH = getenv("BIRCH_LIBRARY_PATH");
  std::string input;

  if (BIRCH_UNIT) {
    unit = BIRCH_UNIT;
  }
  if (BIRCH_PRECISION) {
    precision = BIRCH_PRECISION;
  }
  if (BIRCH_BACKEND) {
    backend = BIRCH_BACKEND;
  }
  #ifdef PREFIX
  prefix = STRINGIFY(PREFIX);
  #endif
  if (BIRCH_PREFIX) {
    prefix = BIRCH_PREFIX;
  }

  /* share dirs */
  if (BIRCH_SHARE_PATH) {
    std::stringstream birch_share_path(BIRCH_SHARE_PATH);
    while (std::getline(birch_share_path, input, ':')) {
      shareDirs.push_back(input);
    }
  }
  #ifdef DATADIR
  shareDirs.push_back(fs::path(STRINGIFY(DATADIR)) / "birch");
  #endif
  if (!prefix.empty()) {
    shareDirs.push_back(fs::path(prefix) / "share" / "birch");
  }
  #ifdef __APPLE__
  /* Homebrew may install to /opt/homebrew on an M1 Mac, which seems not to be
   * reflected in PREFIX; workaround for now */
  shareDirs.push_back(fs::path("/") / "opt" / "homebrew" / "share" / "birch");
  #endif

  /* include dirs */
  if (BIRCH_INCLUDE_PATH) {
    std::stringstream birch_include_path(BIRCH_INCLUDE_PATH);
    while (std::getline(birch_include_path, input, ':')) {
      includeDirs.push_back(input);
    }
  }
  #ifdef INCLUDEDIR
  includeDirs.push_back(STRINGIFY(INCLUDEDIR));
  #endif
  if (!prefix.empty()) {
    includeDirs.push_back(fs::path(prefix) / "include");
  }
  #ifdef __APPLE__
  /* Homebrew may install to /opt/homebrew on an M1 Mac, which seems not to be
   * reflected in PREFIX; workaround for now */
  includeDirs.push_back(fs::path("/") / "opt" / "homebrew" / "include");
  #endif
  /* similarly on Mac, but perhaps elsewhere, these standard paths may not be
   * included, despite PREFIX */
  includeDirs.push_back(fs::path("/") / "usr" / "local" / "include");
  includeDirs.push_back(fs::path("/") / "usr" / "include");

  /* lib dirs */
  if (BIRCH_LIBRARY_PATH) {
    std::stringstream birch_library_path(BIRCH_LIBRARY_PATH);
    while (std::getline(birch_library_path, input, ':')) {
      libDirs.push_back(input);
    }
  }
  #ifdef LIBDIR
  libDirs.push_back(STRINGIFY(LIBDIR));
  #endif
  if (!prefix.empty()) {
    if (fs::exists(fs::path(prefix) / "lib64")) {
      libDirs.push_back(fs::path(prefix) / "lib64");
    }
    if (fs::exists(fs::path(prefix) / "lib")) {
      libDirs.push_back(fs::path(prefix) / "lib");
    }
  }
  #ifdef __APPLE__
  /* Homebrew may install to /opt/homebrew on an M1 Mac, which seems not to be
   * reflected in PREFIX; workaround for now */
  if (fs::exists(fs::path("/") / "opt" / "homebrew" / "lib")) {
    libDirs.push_back(fs::path(fs::path("/") / "opt" / "homebrew" / "lib"));
  }
  #endif

  /* command-line options */
  enum {
    PACKAGE_ARG = 256,
    PREFIX_ARG,
    ARCH_ARG,
    UNIT_ARG,
    MODE_ARG,
    PRECISION_ARG,
    BACKEND_ARG,
    JOBS_ARG,
    ENABLE_TEST_ARG,
    DISABLE_TEST_ARG,
    ENABLE_RELEASE_ARG,
    DISABLE_RELEASE_ARG,
    ENABLE_SINGLE_ARG,
    DISABLE_SINGLE_ARG,
    ENABLE_DOUBLE_ARG,
    DISABLE_DOUBLE_ARG,
    ENABLE_STATIC_ARG,
    DISABLE_STATIC_ARG,
    ENABLE_SHARED_ARG,
    DISABLE_SHARED_ARG,
    ENABLE_STANDALONE_ARG,
    DISABLE_STANDALONE_ARG,
    ENABLE_OPENMP_ARG,
    DISABLE_OPENMP_ARG,
    ENABLE_ASSERT_ARG,
    DISABLE_ASSERT_ARG,
    ENABLE_OPTIMIZE_ARG,
    DISABLE_OPTIMIZE_ARG,
    ENABLE_DEBUG_ARG,
    DISABLE_DEBUG_ARG,
    ENABLE_COVERAGE_ARG,
    DISABLE_COVERAGE_ARG,
    ENABLE_WARNINGS_ARG,
    DISABLE_WARNINGS_ARG,
    ENABLE_NOTES_ARG,
    DISABLE_NOTES_ARG,
    ENABLE_TRANSLATE_ARG,
    DISABLE_TRANSLATE_ARG,
    ENABLE_VERBOSE_ARG,
    DISABLE_VERBOSE_ARG
  };

  int c, option_index;
  option long_options[] = {
      { "package", required_argument, 0, PACKAGE_ARG },
      { "prefix", required_argument, 0, PREFIX_ARG },
      { "arch", required_argument, 0, ARCH_ARG },
      { "unit", required_argument, 0, UNIT_ARG },
      { "mode", required_argument, 0, MODE_ARG },
      { "precision", required_argument, 0, PRECISION_ARG },
      { "backend", required_argument, 0, BACKEND_ARG },
      { "jobs", required_argument, 0, JOBS_ARG },
      { "enable-test", no_argument, 0, ENABLE_TEST_ARG },
      { "disable-test", no_argument, 0, DISABLE_TEST_ARG },
      { "enable-release", no_argument, 0, ENABLE_RELEASE_ARG },
      { "disable-release", no_argument, 0, DISABLE_RELEASE_ARG },
      { "enable-single", no_argument, 0, ENABLE_SINGLE_ARG },
      { "disable-single", no_argument, 0, DISABLE_SINGLE_ARG },
      { "enable-double", no_argument, 0, ENABLE_DOUBLE_ARG },
      { "disable-double", no_argument, 0, DISABLE_DOUBLE_ARG },
      { "enable-static", no_argument, 0, ENABLE_STATIC_ARG },
      { "disable-static", no_argument, 0, DISABLE_STATIC_ARG },
      { "enable-shared", no_argument, 0, ENABLE_SHARED_ARG },
      { "disable-shared", no_argument, 0, DISABLE_SHARED_ARG },
      { "enable-standalone", no_argument, 0, ENABLE_STANDALONE_ARG },
      { "disable-standalone", no_argument, 0, DISABLE_STANDALONE_ARG },
      { "enable-openmp", no_argument, 0, ENABLE_OPENMP_ARG },
      { "disable-openmp", no_argument, 0, DISABLE_OPENMP_ARG },
      { "enable-assert", no_argument, 0, ENABLE_ASSERT_ARG },
      { "disable-assert", no_argument, 0, DISABLE_ASSERT_ARG },
      { "enable-optimize", no_argument, 0, ENABLE_OPTIMIZE_ARG },
      { "disable-optimize", no_argument, 0, DISABLE_OPTIMIZE_ARG },
      { "enable-debug", no_argument, 0, ENABLE_DEBUG_ARG },
      { "disable-debug", no_argument, 0, DISABLE_DEBUG_ARG },
      { "enable-coverage", no_argument, 0, ENABLE_COVERAGE_ARG },
      { "disable-coverage", no_argument, 0, DISABLE_COVERAGE_ARG },
      { "enable-warnings", no_argument, 0, ENABLE_WARNINGS_ARG },
      { "disable-warnings", no_argument, 0, DISABLE_WARNINGS_ARG },
      { "enable-notes", no_argument, 0, ENABLE_NOTES_ARG },
      { "disable-notes", no_argument, 0, DISABLE_NOTES_ARG },
      { "enable-translate", no_argument, 0, ENABLE_TRANSLATE_ARG },
      { "disable-translate", no_argument, 0, DISABLE_TRANSLATE_ARG },
      { "enable-verbose", no_argument, 0, ENABLE_VERBOSE_ARG },
      { "disable-verbose", no_argument, 0, DISABLE_VERBOSE_ARG },
      { 0, 0, 0, 0 }
  };
  const char* short_options = "-";  // treats non-options as short option 1

  /* mutable copy of argv and argc */
  largv.insert(largv.begin(), argv, argv + argc);

  std::vector<char*> unknown;
  opterr = 0;  // handle error reporting ourselves
  c = getopt_long_only(largv.size(), largv.data(), short_options,
      long_options, &option_index);
  while (c != -1) {
    switch (c) {
    case PACKAGE_ARG:
      packageName = optarg;
      break;
    case PREFIX_ARG:
      prefix = optarg;
      break;
    case ARCH_ARG:
      arch = optarg;
      break;
    case UNIT_ARG:
      unit = optarg;
      break;
    case MODE_ARG:
      warn("--mode is deprecated and will be removed in future: there are no longer separate debug and release builds.");
      break;
    case PRECISION_ARG:
      precision = optarg;
      break;
    case BACKEND_ARG:
      backend = optarg;
      break;
    case JOBS_ARG:
      jobs = atoi(optarg);
      break;
    case ENABLE_TEST_ARG:
      warn("--enable-test is deprecated and will be removed in future: setting --enable-coverage instead.");
      enableCoverage = true;
      break;
    case DISABLE_TEST_ARG:
      warn("--disable-test is deprecated and will be removed in future: setting --disable-coverage instead.");
      enableCoverage = false;
      break;
    case ENABLE_RELEASE_ARG:
      warn("--enable-release is deprecated and will be removed in future: setting --enable-optimize instead.");
      enableOptimize = true;
      break;
    case DISABLE_RELEASE_ARG:
      warn("--disable-release is deprecated and will be removed in future: setting --disable-optimize instead.");
      enableOptimize = false;
      break;
    case ENABLE_SINGLE_ARG:
      enableSingle = true;
      break;
    case DISABLE_SINGLE_ARG:
      enableSingle = false;
      break;
    case ENABLE_DOUBLE_ARG:
      enableDouble = true;
      break;
    case DISABLE_DOUBLE_ARG:
      enableDouble = false;
      break;
    case ENABLE_STATIC_ARG:
      enableStatic = true;
      break;
    case DISABLE_STATIC_ARG:
      enableStatic = false;
      break;
    case ENABLE_SHARED_ARG:
      enableShared = true;
      break;
    case DISABLE_SHARED_ARG:
      enableShared = false;
      break;
    case ENABLE_STANDALONE_ARG:
      enableStandalone = true;
      break;
    case DISABLE_STANDALONE_ARG:
      enableStandalone = false;
      break;
    case ENABLE_OPENMP_ARG:
      enableOpenmp = true;
      break;
    case DISABLE_OPENMP_ARG:
      enableOpenmp = false;
      break;
    case ENABLE_ASSERT_ARG:
      enableAssert = true;
      break;
    case DISABLE_ASSERT_ARG:
      enableAssert = false;
      break;
    case ENABLE_OPTIMIZE_ARG:
      enableOptimize = true;
      break;
    case DISABLE_OPTIMIZE_ARG:
      enableOptimize = false;
      break;
    case ENABLE_DEBUG_ARG:
      enableDebug = true;
      break;
    case DISABLE_DEBUG_ARG:
      enableDebug = false;
      break;
    case ENABLE_COVERAGE_ARG:
      enableCoverage = true;
      break;
    case DISABLE_COVERAGE_ARG:
      enableCoverage = false;
      break;
    case ENABLE_WARNINGS_ARG:
      enableWarnings = true;
      break;
    case DISABLE_WARNINGS_ARG:
      enableWarnings = false;
      break;
    case ENABLE_NOTES_ARG:
      enableNotes = true;
      break;
    case DISABLE_NOTES_ARG:
      enableNotes = false;
      break;
    case ENABLE_TRANSLATE_ARG:
      enableTranslate = true;
      break;
    case DISABLE_TRANSLATE_ARG:
      enableTranslate = false;
      break;
    case ENABLE_VERBOSE_ARG:
      enableVerbose = true;
      break;
    case DISABLE_VERBOSE_ARG:
      enableVerbose = false;
      break;
    case '?':  // unknown option
    case 1:  // not an option
      unknown.push_back(largv[optind - 1]);
      largv.erase(largv.begin() + optind - 1, largv.begin() + optind);
      --optind;
      break;
    }
    c = getopt_long_only(largv.size(), largv.data(), short_options,
        long_options, &option_index);
  }
  largv.insert(largv.end(), unknown.begin(), unknown.end());

  /* some error checking */
  if (jobs <= 0) {
    throw DriverException("--jobs must be a positive integer.");
  }
  if (!arch.empty() && arch != "native") {
    throw DriverException("--arch must be native, or empty.");
  }
  if (unit != "unity" && unit != "dir" && unit != "file") {
    throw DriverException("--unit must be unity, dir, or file.");
  }
  if (precision != "single" && precision != "double") {
    throw DriverException("--precision must be single or double.");
  }
  if (backend != "eigen" && backend != "cuda") {
    throw DriverException("--backend must be eigen or cuda.");
  }
}

std::string birch::Driver::numbirch() {
  std::string name = "libnumbirch";
  if (backend.compare("eigen") != 0) {
    name += "-" + backend;
  }
  if (precision.compare("single") == 0) {
    name += "-single";
  }
  fs::path lib = name;
  #ifdef __APPLE__
  lib.replace_extension(".dylib");
  #else
  lib.replace_extension(".so");
  #endif
  try {
    return find(libDirs, lib);
  } catch (const FileNotFoundException& e) {
    return lib;  // let dlopen() search standard paths
  }
}

std::string birch::Driver::library() {
  /* get package information */
  try {
    /* load the package meta information, if indeed there is any, otherwise
     * this will throw an exception which is caught below */
    meta();

    /* also in this case, search the .libs directory first for a local build
     * of the package shared library */
    libDirs.push_front(".libs");
  } catch (DriverException) {
    // probably not running in a package directory, but can use installed
    // libraries instead
  }

  /* name of the shared library file we expect to find */
  std::string name = "lib" + tar(packageName);
  if (precision.compare("single") == 0) {
    name += "-single";
  }
  fs::path lib = name;
  #ifdef __APPLE__
  lib.replace_extension(".dylib");
  #else
  lib.replace_extension(".so");
  #endif
  try {
  return find(libDirs, lib);
  } catch (const FileNotFoundException& e) {
    return lib;  // let dlopen() search standard paths
  }
}

int birch::Driver::argc() {
  return largv.size();
}

char** birch::Driver::argv() {
  return largv.data();
}

void birch::Driver::bootstrap() {
  meta();
  setup();
  transpile();

  if (newBootstrap || newConfigure || newMake || !fs::exists("configure")) {
    std::stringstream cmd;
    cmd << (fs::path(".") / "bootstrap");
    if (enableVerbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > bootstrap.log 2>&1";
    }

    int status = std::system(cmd.str().c_str());
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::stringstream buf;
      buf << "bootstrap failed";
      if (!enableVerbose) {
        buf << ", see bootstrap.log for details, perhaps run `birch clean` and try again";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void birch::Driver::configure() {
  bootstrap();

  if (newBootstrap || newConfigure || newMake || !fs::exists("Makefile")) {
    /* compile and link flags */
    std::stringstream cppflags, cflags, cxxflags, ldflags, options, cmd;
    if (arch == "native") {
      cflags << " -march=native";
      cxxflags << " -march=native";
    }
    if (enableOptimize) {
      cflags << " -O3 -flto";
      cxxflags << " -O3 -flto";
    }
    if (enableDebug) {
      cflags << " -g";
      cxxflags << " -g";
    }
    if (enableCoverage) {
      cflags << " -fno-inline --coverage";
      cxxflags << "  -fno-inline --coverage";
    }
    if (enableWarnings) {
      cppflags << " -Wall";
      cflags << " -Wall";
      cxxflags << " -Wall";
    }

    /* include path */
    for (auto iter = includeDirs.begin(); iter != includeDirs.end();
        ++iter) {
      cppflags << " -I" << iter->string();
    }
    for (auto iter = libDirs.begin(); iter != libDirs.end(); ++iter) {
      ldflags << " -L" << iter->string();
    }

    /* library path */
    for (auto iter = libDirs.begin(); iter != libDirs.end(); ++iter) {
      ldflags << " -Wl,-rpath," << iter->string();
    }

    /* configure options */
    if (enableSingle) {
      options << " --enable-single";
    } else {
      options << " --disable-single";
    }
    if (enableDouble) {
      options << " --enable-double";
    } else {
      options << " --disable-double";
    }
    if (enableStatic) {
      options << " --enable-static";
    } else {
      options << " --disable-static";
    }
    if (enableShared) {
      options << " --enable-shared";
    } else {
      options << " --disable-shared";
    }
    if (enableStandalone) {
      options << " --enable-standalone";
    } else {
      options << " --disable-standalone";
    }
    if (enableOpenmp) {
      options << " --enable-openmp";
    } else {
      options << " --disable-openmp";
    }
    if (enableAssert) {
      options << " --enable-assert";
    } else {
      options << " --disable-assert";
    }

    if (!prefix.empty()) {
      options << " --prefix=" << prefix;
    }
    options << " --config-cache";
    options << " INSTALL=\"install -p\"";
    if (!cppflags.str().empty()) {
      options << " CPPFLAGS=\"$CPPFLAGS " << cppflags.str() << "\"";
    }
    if (!cflags.str().empty()) {
      options << " CFLAGS=\"$CFLAGS " << cflags.str() << "\"";
    }
    if (!cxxflags.str().empty()) {
      options << " CXXFLAGS=\"$CXXFLAGS " << cxxflags.str() << "\"";
    }
    if (!ldflags.str().empty()) {
      options << " LDFLAGS=\"$LDFLAGS " << ldflags.str() << "\"";
    }

    /* command */
    if (arch == "js" || arch == "wasm") {
      cmd << "emconfigure ";
    }
    cmd << (fs::path(".") / "configure") << ' ' << options.str();
    if (enableVerbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > configure.log 2>&1";
    }

    int status = std::system(cmd.str().c_str());
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::stringstream buf;
      buf << "configure failed";
      if (!enableVerbose) {
        buf << ", see configure.log and config.log for details, perhaps run `birch clean` and try again";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void birch::Driver::build() {
  configure();
  target();
}

void birch::Driver::install() {
  configure();
  target("install");
}

void birch::Driver::uninstall() {
  configure();
  target("uninstall");
}

void birch::Driver::dist() {
  meta();

  /* determine archive name, format 'name-version' */
  auto archive = tar(packageName) + "-" + packageVersion;

  /* archiving command */
  std::stringstream cmd;
  #if defined(__APPLE__) || defined(__FreeBSD__)
  /* BSD tar */
  cmd << "tar -s '#^#" << archive << "/#' -czf " << archive << ".tar.gz";
  #else
  /* assume GNU tar */
  cmd << "tar -czf " << archive << ".tar.gz --transform='s/^/" << archive << "\\//'";
  #endif
  for (auto key : {
      "manifest.header",
      "manifest.source",
      "manifest.data",
      "manifest.other"}) {
    for (auto file : metaFiles[key]) {
      cmd << ' ' << file;
    }
  }
  if (enableVerbose) {
    std::cerr << cmd.str() << std::endl;
  } else {
    cmd << " > tar.log 2>&1";
  }

  /* run command */
  int status = std::system(cmd.str().c_str());
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::stringstream buf;
    buf << "tar failed";
    if (!enableVerbose) {
      buf << ", see tar.log for details, perhaps run `birch clean` and try again";
    }
    buf << '.';
    throw DriverException(buf.str());
  }
}

void birch::Driver::clean() {
  meta();
  auto tarName = tar(packageName);
  auto canonicalName = canonical(packageName);

  fs::remove_all("build");
  fs::remove_all("autom4te.cache");
  fs::remove_all("m4");
  fs::remove_all(".deps");
  fs::remove_all(".libs");
  fs::remove("aclocal.m4");
  fs::remove("bootstrap.log");
  fs::remove("bootstrap");
  fs::remove("compile");
  fs::remove("config.cache");
  fs::remove("config.guess");
  fs::remove("config.log");
  fs::remove("config.status");
  fs::remove("config.sub");
  fs::remove("configure");
  fs::remove("configure.ac");
  fs::remove("configure.log");
  fs::remove("depcomp");
  fs::remove("install-sh");
  fs::remove("libtool");
  fs::remove("ltmain.sh");
  fs::remove("Makefile");
  fs::remove("Makefile.am");
  fs::remove("Makefile.in");
  fs::remove("missing");
  fs::remove(tarName + ".cpp");
  fs::remove(tarName + ".hpp");

  /* library artifacts */
  fs::remove("lib" + canonicalName + "_la" + tarName + ".lo");
  fs::remove("lib" + canonicalName + "_single_la" + tarName + ".lo");
  fs::remove("lib" + tarName + ".la");
  fs::remove("lib" + tarName + "-single.la");

  /* standalone program artifacts */
  fs::remove(canonicalName + '-' + tarName + "-standalone.o");
  fs::remove(canonicalName + "_single-" + tarName + "-standalone.o");
  fs::remove(tarName);
  fs::remove(tarName + "-single");
  fs::remove(tarName + "-standalone.cpp");
  fs::remove(tarName + "-standalone.o");

  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    fs::path source = tarName;
    source.replace_extension(".cpp");
    fs::remove(source);

    fs::path object = source.parent_path() / ("lib" + canonicalName +
        "_la-" + source.filename().string());
    object.replace_extension(".lo");
    fs::remove(object);
    object.replace_extension(".o");
    fs::remove(object);

  } else if (unit == "file") {
    /* sources go into one *.cpp file for each *.birch file */
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".birch") == 0) {
        fs::path libs = file.parent_path() / ".libs";
        fs::remove_all(libs);

        fs::path source = file;
        source.replace_extension(".cpp");
        fs::remove(source);

        fs::path object = source.parent_path() / ("lib" + canonicalName +
            "_la-" + source.filename().string());
        object.replace_extension(".lo");
        fs::remove(object);
        object.replace_extension(".o");
        fs::remove(object);
      }
    }
  } else {
    /* sources go into one *.cpp file for each directory */
    std::unordered_set<std::string> sources;
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".birch") == 0) {
        fs::path libs = file.parent_path() / ".libs";
        fs::remove_all(libs);

        fs::path source = file.parent_path() / tarName;
        source.replace_extension(".cpp");
        fs::remove(source);
        source.replace_extension(".lo");

        fs::path object;
        object = source.parent_path() / ("lib" + canonicalName + "_la-" + source.filename().string());
        fs::remove(object);
      }
    }
  }
}

void birch::Driver::init() {
  fs::create_directory("src");
  fs::create_directory("config");
  fs::create_directory("input");
  fs::create_directory("output");

  copy_with_prompt(find(shareDirs, "gitignore"), ".gitignore");
  copy_with_prompt(find(shareDirs, "LICENSE"), "LICENSE");
  copy_with_prompt(find(shareDirs, "hello.birch"), fs::path("src") / "hello.birch");

  if (copy_with_prompt(find(shareDirs, "birch.yml"), "birch.yml")) {
    replace_tag("birch.yml", "PACKAGE_NAME", packageName);
  }
  if (copy_with_prompt(find(shareDirs, "README.md"), "README.md")) {
    replace_tag("README.md", "PACKAGE_NAME", packageName);
  }
  if (copy_with_prompt(find(shareDirs, "mkdocs.yml"), "mkdocs.yml")) {
    replace_tag("mkdocs.yml", "PACKAGE_NAME", packageName);
  }
}

void birch::Driver::audit() {
  meta();

  /* check LICENSE */
  if (!fs::exists("LICENSE")) {
    warn("no LICENSE file; create a LICENSE file containing the "
        "distribution license (e.g. GPL or BSD) of the package.");
  } else if (allFiles.find("LICENSE") == allFiles.end()) {
    warn("LICENSE file is not listed in build configuration.");
  }

  /* check README.md */
  if (!fs::exists("README.md")) {
    warn("no README.md file; create a README.md file documenting the "
        "package in Markdown format.");
  } else if (allFiles.find("README.md") == allFiles.end()) {
    warn("README.md file is not listed in build configuration.");
  }

  /* check for files that might be missing from meta */
  std::unordered_set<std::string> interesting, exclude;

  interesting.insert(".birch");
  interesting.insert(".sh");
  interesting.insert(".yaml");
  interesting.insert(".yml");
  interesting.insert(".json");

  exclude.insert("bootstrap");
  exclude.insert("ltmain.sh");

  fs::recursive_directory_iterator iter("."), end;
  while (iter != end) {
    auto path = remove_first(iter->path());
    auto name = path.filename().string();
    auto ext = path.extension().string();
    if (path.string() == "build" || path.string() == "output"
        || path.string() == "site") {
      iter.FS_DISABLE_RECURSION_PENDING();
    } else if (interesting.find(ext) != interesting.end()
        && exclude.find(name) == exclude.end()) {
      if (allFiles.find(path.string()) == allFiles.end()) {
        warn(std::string("is ") + path.string()
                + " missing from build configuration?");
      }
    }
    ++iter;
  }
}

void birch::Driver::docs() {
  meta();
  Package* package = createPackage();

  /* parse all files */
  Compiler compiler(package, unit);
  compiler.parse();

  /* output everything into single file */
  fs_stream::ofstream docsStream("DOCS.md");
  if (docsStream.fail()) {
    std::stringstream buf;
    buf << "Could not open DOCS.md for writing.";
    throw DriverException(buf.str());
  }
  MarkdownGenerator output(docsStream);
  output << package;
  docsStream.close();

  /* split that file into multiple files for mkdocs */
  fs::path docs("docs"), file;
  fs::create_directories(docs);
  fs::create_directories(docs / "types");
  fs::create_directories(docs / "variables");
  fs::create_directories(docs / "programs");
  fs::create_directories(docs / "functions");
  fs::create_directories(docs / "operators");
  fs::create_directories(docs / "classes");
  fs::create_directories(docs / "structs");

  /* index file */
  if (fs::exists("README.md")) {
    copy_with_force("README.md", docs / "index.md");
  } else {
    docsStream.open(docs / "index.md");
    docsStream << packageDescription << '\n';
    docsStream.close();
  }

  std::string str = read_all("DOCS.md");
  std::regex reg("(?:^|\r?\n)(##?) (.*?)(?=\r?\n|$)");
  std::smatch match;
  std::string str1 = str, h1, h2;
  while (std::regex_search(str1, match, reg)) {
    if (docsStream.is_open()) {
      docsStream << match.prefix();
    }
    if (match.str(1) == "#") {
      /* first level header */
      h1 = match.str(2);

      /* among first-level headers, only variables have their own page, rather
       * than being further split into a page per item */
      if (h1 == "Variables") {
        std::string dir = lower(h1);
        file = fs::path(dir) / "index.md";
        if (docsStream.is_open()) {
          docsStream.close();
        }
        docsStream.open(docs / file);
        docsStream << "# " << h1 << "\n\n";
      }
      h1 = lower(h1);
      h1 = std::regex_replace(h1, std::regex(" "), "_");
    } else {
      /* second level header */
      h2 = match.str(2);
      file = fs::path(nice(h1)) / (nice(h2) + ".md");
      if (docsStream.is_open()) {
        docsStream.close();
      }
      docsStream.open(docs / file);
      docsStream << "title: " << h2 << "\n";
      docsStream << "---\n\n";
    }
    str1 = match.suffix();
  }
  if (docsStream.is_open()) {
    docsStream << str1;
    docsStream.close();
  }
}

void birch::Driver::help() {
  std::cout << std::endl;
  if (largv.size() >= 2) {
    std::string command = largv.at(1);
    if (command.compare("init") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch init --package Name" << std::endl;
      std::cout << std::endl;
      std::cout << "Initialize the working directory for a new package." << std::endl;
      std::cout << std::endl;
      std::cout << "  --package (default Untitled): Name of the package." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/init/" << std::endl;
    } else if (command.compare("audit") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch audit" << std::endl;
      std::cout << std::endl;
      std::cout << "Audit the package for possible issues." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/audit/" << std::endl;
    } else if (command.compare("bootstrap") == 0 ||
        command.compare("configure") == 0 ||
        command.compare("build") == 0 ||
        command.compare("install") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch bootstrap" << std::endl;
      std::cout << "  birch configure [options...]" << std::endl;
      std::cout << "  birch build [options...]" << std::endl;
      std::cout << "  birch install [options...]" << std::endl;
      std::cout << std::endl;
      std::cout << "Build the package up to the given stage (in order: bootstrap, configure, build," << std::endl;
      std::cout << "install)." << std::endl;
      std::cout << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-single / --disable-single (default enabled):" << std::endl;
      std::cout << "  Enable/disable single precision floating point build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-double / --disable-double (default enabled):" << std::endl;
      std::cout << "  Enable/disable double precision floating point build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-static / --disable-static (default enabled):" << std::endl;
      std::cout << "  Enable/disable static library build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-shared / --disable-shared (default enabled):" << std::endl;
      std::cout << "  Enable/disable shared library build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-standalone / --disable-standalone (default disabled):" << std::endl;
      std::cout << "  Enable/disable standalone program build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-openmp / --disable-openmp (default enabled):" << std::endl;
      std::cout << "  Enable/disable OpenMP support." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-assert / --disable-assert (default enabled):" << std::endl;
      std::cout << "  Enable/disable assertion checks." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-optimize / --disable-optimize (default enabled):" << std::endl;
      std::cout << "  Enable/disable optimizations." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-debug / --disable-debug (default enabled):" << std::endl;
      std::cout << "  Enable/disable debugging information." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-coverage / --disable-coverage (default disabled):" << std::endl;
      std::cout << "  Enable/disable coverage information." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-warnings / --disable-warnings (default enabled):" << std::endl;
      std::cout << "  Enable/disable compiler warnings." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-notes / --disable-notes (default enabled):" << std::endl;
      std::cout << "  Enable/disable compiler notes." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-translate / --disable-translate (default enabled):" << std::endl;
      std::cout << "  Enable/disable approximate translation of C++ compiler messages to Birch." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-verbose / --disable-verbose (default disabled):" << std::endl;
      std::cout << "  Show all compiler output." << std::endl;
      std::cout << std::endl;
      std::cout << "  --unit (default `dir`, valid values `unity`, `dir`, `file`):" << std::endl;
      std::cout << "  Set the compile unit to use when transpiling Birch to C++." << std::endl;
      std::cout << std::endl;
      std::cout << "  --prefix (default imputed):" << std::endl;
      std::cout << "  Installation prefix. Defaults to the same prefix used when installing the birch" << std::endl;
      std::cout << "  driver program." << std::endl;
      std::cout << std::endl;
      std::cout << "  --precision (default `double`, valid values `single`, `double`):" << std::endl;
      std::cout << "  Set the floating point precision of the build to run." << std::endl;
      std::cout << std::endl;
      std::cout << "  --backend (default `eigen`, valid values `eigen`, `cuda`):" << std::endl;
      std::cout << "  Set the NumBirch backend to run." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/bootstrap/" << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/configure/" << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/build/" << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/install/" << std::endl;
    } else if (command.compare("uninstall") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch uninstall" << std::endl;
      std::cout << std::endl;
      std::cout << "Uninstall the package." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/uninstall/" << std::endl;
    } else if (command.compare("dist") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch dist" << std::endl;
      std::cout << std::endl;
      std::cout << "Build the package source tarball." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/dist/" << std::endl;
    } else if (command.compare("docs") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch docs" << std::endl;
      std::cout << std::endl;
      std::cout << "Build the package documentation." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/docs/" << std::endl;
    } else if (command.compare("clean") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch clean" << std::endl;
      std::cout << std::endl;
      std::cout << "Remove build files." << std::endl;
      std::cout << std::endl;
      std::cout << "More information is available at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://docs.birch.sh/libraries/Standard/programs/clean/" << std::endl;
    } else if (command.compare("help") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch help" << std::endl;
      std::cout << "  birch help <command>" << std::endl;
      std::cout << std::endl;
      std::cout << "Print the general help message, or the help message for <command>." << std::endl;
    } else {
      std::cout << "Command " << largv.at(1) << " is not a recognized command."  << std::endl;
    }
  } else {
    std::cout << "Usage:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch <command> [options...]" << std::endl;
    std::cout << std::endl;
    std::cout << "where <command> is one of:" << std::endl;
    std::cout << std::endl;
    std::cout << "  init          Initialize the working directory for a new package." << std::endl;
    std::cout << "  audit         Audit the package for common issues." << std::endl;
    std::cout << "  bootstrap     Bootstrap the package, creating build files." << std::endl;
    std::cout << "  configure     Bootstrap and configure the package." << std::endl;
    std::cout << "  build         Bootstrap, configure and build the package." << std::endl;
    std::cout << "  install       Bootstrap, configure, build and install the package." << std::endl;
    std::cout << "  uninstall     Uninstall the package." << std::endl;
    std::cout << "  dist          Build the package source tarball." << std::endl;
    std::cout << "  docs          Build the package documentation." << std::endl;
    std::cout << "  clean         Remove build files." << std::endl;
    std::cout << "  help          Print this help message." << std::endl;
    std::cout << std::endl;
    std::cout << "and available [options...] for <command> can be found with:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch help <command>" << std::endl;
    std::cout << std::endl;
    std::cout << "To call a program defined in the package use:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch <program> [options...]" << std::endl;
    std::cout << std::endl;
    std::cout << "More information is available at:" << std::endl;
    std::cout << std::endl;
    std::cout << "  https://docs.birch.sh/" << std::endl;
  }
  std::cout << std::endl;
}

void birch::Driver::meta() {
  /* clear any previous read */
  metaContents.clear();
  metaFiles.clear();
  allFiles.clear();

  /* parse build configuration file */
  MetaParser parser;
  metaContents = parser.parse();
  if (!metaContents["name"].empty()) {
    packageName = metaContents["name"].front();
  }
  if (!metaContents["description"].empty()) {
    packageDescription = metaContents["description"].front();
  }
  if (!metaContents["version"].empty()) {
    packageVersion = metaContents["version"].front();
  }

  /* update some old names */
  for (auto& value : metaContents["require.package"]) {
    if (value == "Standard" || value == "Cairo" || value == "SQLite") {
      warn(std::string("package ") + value + " is now named Birch." + value +
          ", update package meta file.");
      value = std::string("Birch.") + value;
    }
  }

  /* check manifest files */
  readFiles("manifest.header");
  readFiles("manifest.source");
  readFiles("manifest.data");
  readFiles("manifest.other");
}

void birch::Driver::setup() {
  auto tarName = tar(packageName);
  auto canonicalName = canonical(packageName);

  /* copy source files */
  copy_if_newer(find(shareDirs, "standalone.cpp"),
      tarName + "-standalone.cpp");

  /* copy build files */
  newBootstrap = copy_if_newer(find(shareDirs, "bootstrap"), "bootstrap");
  fs::permissions("bootstrap", fs::status("bootstrap").permissions()|
      FS_OWNER_EXE);

  auto m4_dir = fs::path("m4");
  if (!fs::exists(m4_dir)) {
    if (!fs::create_directory(m4_dir)) {
      std::stringstream buf;
      buf << "Could not create m4 directory " << m4_dir << '.';
      throw DriverException(buf.str());
    }
  }
  copy_if_newer(find(shareDirs, "ax_check_compile_flag.m4"),
      m4_dir / "ax_check_compile_flag.m4");
  copy_if_newer(find(shareDirs, "ax_check_define.m4"),
      m4_dir / "ax_check_define.m4");
  copy_if_newer(find(shareDirs, "ax_cxx_compile_stdcxx.m4"),
      m4_dir / "ax_cxx_compile_stdcxx.m4");
  copy_if_newer(find(shareDirs, "ax_gcc_builtin.m4"),
      m4_dir / "ax_gcc_builtin.m4");

  /* configure.ac */
  std::string contents = read_all(find(shareDirs, "configure.ac"));
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_NAME\\}\\}"), packageName);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_VERSION\\}\\}"), packageVersion);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_TARNAME\\}\\}"), tarName);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_CANONICAL_NAME\\}\\}"), canonicalName);
  std::stringstream configureStream;
  configureStream << contents << "\n\n";
  configureStream << "if $checks; then\n";

  /* required headers */
  for (auto value : metaContents["require.header"]) {
    configureStream << "  AC_CHECK_HEADERS([" << value << "], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])\n";
  }
  for (auto value : metaContents["require.package"]) {
    auto tarName = tar(value);
    configureStream << "  AC_CHECK_HEADERS([" << tarName << ".hpp], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])\n";
  }

  /* required libraries */
  for (auto value : metaContents["require.library"]) {
    configureStream << "  AC_CHECK_LIB([" << value << "], [main], [], [AC_MSG_ERROR([required library not found.])])\n";
  }
  for (auto value : metaContents["require.package"]) {
    auto tarName = tar(value);
    configureStream << "  if $single; then\n";
    configureStream << "    AC_CHECK_LIB([" << tarName << "-single], [main], [SINGLE_LIBS=\"$SINGLE_LIBS -l" << tarName << "-single\"], [AC_MSG_ERROR([required library not found.])], [$SINGLE_LIBS])\n";
    configureStream << "  fi\n";
    configureStream << "  if $double; then\n";
    configureStream << "    AC_CHECK_LIB([" << tarName << "], [main], [DOUBLE_LIBS=\"$DOUBLE_LIBS -l" << tarName << "\"], [AC_MSG_ERROR([required library not found.])], [$DOUBLE_LIBS])\n";
    configureStream << "  fi\n";
  }

  /* required programs */
  for (auto value : metaContents["require.program"]) {
    configureStream << "  AC_PATH_PROG([PROG], [" << value << "], [])\n";
    configureStream << "  if test \"$PROG\" = \"\"; then\n";
    configureStream << "    AC_MSG_ERROR([required program not found.])\n";
    configureStream << "  fi\n";
  }
  configureStream << "fi\n";

  /* footer */
  configureStream << "AC_SUBST([DOUBLE_LIBS])\n";
  configureStream << "AC_SUBST([SINGLE_LIBS])\n";
  configureStream << "AC_SUBST([STANDALONE_LIBS])\n";
  configureStream << "\n";
  configureStream << "AC_CONFIG_FILES([Makefile])\n";
  configureStream << "AC_OUTPUT\n";

  newConfigure = write_all_if_different("configure.ac",
      configureStream.str());

  /* Makefile.am */
  contents = read_all(find(shareDirs, "Makefile.am"));
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_NAME\\}\\}"), packageName);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_VERSION\\}\\}"), packageVersion);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_TARNAME\\}\\}"), tarName);
  contents = std::regex_replace(contents, std::regex("\\{\\{PACKAGE_CANONICAL_NAME\\}\\}"), canonicalName);

  /* Birch built sources */
  std::stringstream makeStream;
  makeStream << contents << "\n\n";
  makeStream << "COMMON_SOURCES =";
  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    auto source = fs::path(tarName);
    source.replace_extension(".cpp");
    makeStream << " \\\n  " << source.string();
  } else if (unit == "file") {
    /* sources go into one *.cpp file for each *.birch file */
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".birch") == 0) {
        auto source = file;
        source.replace_extension(".cpp");
        makeStream << " \\\n  " << source.string();
      }
    }
  } else {
    /* sources go into one *.cpp file for each directory */
    std::unordered_set<std::string> sources;
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".birch") == 0) {
        auto source = file.parent_path() / tarName;
        source.replace_extension(".cpp");
        if (sources.insert(source.string()).second) {
          makeStream << " \\\n  " << source.string();
        }
      }
    }
  }

  /* C++ sources */
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".cpp") == 0) {
      makeStream << " \\\n  " << file.string();
    }
  }
  makeStream << '\n';

  /* Birch built headers */
  makeStream << "nobase_include_HEADERS =";
  auto header = fs::path(tarName);
  header.replace_extension(".hpp");
  makeStream << " \\\n  " << header.string();

  /* C++ headers */
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".hpp") == 0) {
      makeStream << " \\\n  " << file.string();
    }
  }
  makeStream << '\n';

  /* data files to distribute */
  makeStream << "nobase_dist_pkgdata_DATA = ";
  for (auto file : metaFiles["manifest.data"]) {
    makeStream << " \\\n  " << file.string();
  }
  makeStream << '\n';

  /* other files to distribute */
  makeStream << "dist_noinst_DATA = ";
  for (auto file : metaFiles["manifest.other"]) {
    makeStream << " \\\n  " << file.string();
  }
  makeStream << '\n';

  newMake = write_all_if_different("Makefile.am", makeStream.str());
}

void birch::Driver::transpile() {
  Compiler compiler(createPackage(), unit);
  compiler.parse();
  compiler.gen(enableTranslate);
}

void birch::Driver::target(const std::string& cmd) {
  /* command */
  std::stringstream buf;
  if (arch == "js" || arch == "wasm") {
    buf << "emmake";
  }
  buf << "make";
  if (!enableVerbose) {
    buf << " -s LIBTOOLFLAGS=--silent";
  }

  /* concurrency */
  buf << " -j " << jobs;

  /* target */
  buf << ' ' << cmd << " 2>&1";  // stderr to stdout so pipe has both

  if (enableVerbose) {
    std::cerr << buf.str() << std::endl;
  }

  /* translate the output of make to make it digestable; while we
   * can use pipes to grep and sed within the shell command, this is less
   * portable, and means the command always returns success, even on fail
   * (consider pipefail); instead we use popen instead of system, and process
   * the output with regexes */
  auto options = std::regex_constants::optimize;
  std::string name = "[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩA-Za-z0-9_]+";
  std::string type = name;
  for (auto i = 0; i < 10; ++i) {
    type = name + "(?:<" + type + "(?:, *" + type + ")* *>)?(?:\\[[_,]+\\])?[\\?\\!]*";
  }
  std::string generic = "<" + name + "(?:, *" + name + ")* *>";

  std::regex rxWarnings("warning:", options);
  std::regex rxNotes("note:|required by|required from|\\[with.*?\\]", options);

  std::regex rxSkipLine("In file included from|In function|In member function|In instantiation|instantiation contexts|unrecognized command-line option|^\\s*from|std::enable_if|At global scope:|type_traits", options);
  std::regex rxNamespace("birch::|membirch::|numbirch::", options);
  std::regex rxInternal("\\b(" + name + ")_\\b", options);
  std::regex rxCxxWords("\\b(?:virtual|class|const|typename|template(?= *<))\\b *", options);
  std::regex rxTemplateParameter("template parameter", options);
  std::regex rxTemplateArgument("template argument", options);
  std::regex rxTypeDeduction("before deduction of ‘’", options);
  std::regex rxString("(?:const *)?std::(?:__cxx11::)?basic_string<char>", options);
  std::regex rxReal("\\b(?:double|float)\\b", options);
  std::regex rxInteger("\\bint\\b", options);
  std::regex rxBool("\\bbool\\b", options);
  std::regex rxScalar("Array<(" + type + "), *0 *>", options);
  std::regex rxVector("Array<(" + type + "), *1 *>", options);
  std::regex rxMatrix("Array<(" + type + "), *2 *>", options);
  std::regex rxShared("Shared<(" + type + ") *>", options);
  std::regex rxOptional("std::optional<(" + type + ") *>", options);
  std::regex rxFuture("(?:Future|Scalar)<(" + type + ") *>", options);
  std::regex rxConstRef("(?:const *)?(" + type + ") *&", options);
  std::regex rxThis("\\(\\(" + type + "\\*\\)this\\)->" + type + "::", options);
  std::regex rxAka(" *\\{aka *[^\\}]+?\\}", options);
  std::regex rxValueType("\\busing value_type *= *", options);
  std::regex rxValueType2("::value_type", options);
  std::regex rxDeref("(?:operator)?->", options);
  std::regex rxAssign("‘(?:operator)?=’", options);
  std::regex rxNamespaceSep("::", options);
  std::regex rxAuto("‘auto’", options);
  std::regex rxProgram("In function ‘int ([A-Za-z0-9_]+)\\(int, *char\\*\\*\\)’", options);
  std::regex rxVariable("‘(" + type + ") +(" + name + ")’", options);
  std::regex rxMemberVariable("‘(" + type + ") +(" + type + ")\\.(" + name + ")’", options);
  std::regex rxVoidFunction("‘((?:" + generic + ")?) *void +(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);
  std::regex rxVoidMemberFunction("‘((?:" + generic + ")?) *void +(" + type + ")\\.(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);
  std::regex rxAutoFunction("‘((?:" + generic + ")?) *auto +(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);
  std::regex rxAutoMemberFunction("‘((?:" + generic + ")?) *auto +(" + type + ")\\.(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);
  std::regex rxFunction("‘((?:" + generic + ")?) *(" + type + ") +(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);
  std::regex rxMemberFunction("‘((?:" + generic + ")?) *(" + type + ") +(" + type + ")\\.(" + name + ")\\((.*?)\\)( *\\[with.+\\])?’", options);

  FILE* pipe = popen(buf.str().c_str(), "r");
  if (pipe) {
    char* line = nullptr;
    size_t n = 0;
    while (getline(&line, &n, pipe) > 0) {
      std::string str(line);
      free(line);
      line = nullptr;
      n = 0;

      if (enableTranslate) {
        /* strip namespace and class qualifiers */
        str = std::regex_replace(str, rxNamespace, "");
        str = std::regex_replace(str, rxInternal, "$1");

        /* replace some C++ words */
        str = std::regex_replace(str, rxCxxWords, "");
        str = std::regex_replace(str, rxTemplateParameter, "type parameter");
        str = std::regex_replace(str, rxTemplateArgument, "type argument");
        str = std::regex_replace(str, rxTypeDeduction, "before deduction of return type");

        /* convert back some types */
        str = std::regex_replace(str, rxString, "String");
        str = std::regex_replace(str, rxReal, "Real");
        str = std::regex_replace(str, rxInteger, "Integer");
        str = std::regex_replace(str, rxBool, "Boolean");

        /* replace some types */
        str = std::regex_replace(str, rxScalar, "$1!");
        str = std::regex_replace(str, rxVector, "$1[_]");
        str = std::regex_replace(str, rxMatrix, "$1[_,_]");
        str = std::regex_replace(str, rxConstRef, "$1");
        str = std::regex_replace(str, rxThis, "this.");
        str = std::regex_replace(str, rxAka, "");
        str = std::regex_replace(str, rxValueType, "");
        str = std::regex_replace(str, rxValueType2, "");
        for (auto i = 0; i < 10; ++i) {
          str = std::regex_replace(str, rxOptional, "$1?");
        }
        for (auto i = 0; i < 10; ++i) {
          str = std::regex_replace(str, rxFuture, "$1!");
        }
        for (auto i = 0; i < 10; ++i) {
          str = std::regex_replace(str, rxShared, "$1");
        }

        /* replace some operators */
        str = std::regex_replace(str, rxDeref, ".");
        str = std::regex_replace(str, rxAssign, "‘<-‘");
        str = std::regex_replace(str, rxNamespaceSep, ".");
        str = std::regex_replace(str, rxAuto, "‘let’");

        /* strip suggestions that reveal internal workings */
        str = std::regex_replace(str, rxProgram, "In program ‘$1’");

        /* function and variable declaration syntax */
        str = std::regex_replace(str, rxVariable, "‘$2:$1’");
        str = std::regex_replace(str, rxMemberVariable, "‘$2.$3:$1’");
        str = std::regex_replace(str, rxVoidFunction, "‘$2$1($3)$4’");
        str = std::regex_replace(str, rxVoidMemberFunction, "‘$2.$3$1($4)$5’");
        str = std::regex_replace(str, rxAutoFunction, "‘$2$1($3) -> $4’");
        str = std::regex_replace(str, rxAutoMemberFunction, "‘$2.$3$1($4) -> $5’");
        str = std::regex_replace(str, rxFunction, "‘$3$1($4) -> $2$5’");
        str = std::regex_replace(str, rxMemberFunction, "‘$3.$4$1($5) -> $2$6’");
      }
      if ((enableWarnings || !std::regex_search(str, rxWarnings)) &&
          (enableNotes || !std::regex_search(str, rxNotes)) &&
          (!enableTranslate || !std::regex_search(str, rxSkipLine))) {
        std::cerr << str;
      }
    }
    int status = pclose(pipe);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      throw DriverException("make failed.");
    }
  }
}

birch::Package* birch::Driver::createPackage() {
  Package* package = new Package(packageName);
  for (auto value : metaContents["require.package"]) {
    package->addPackage(value);
  }
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".birch") == 0) {
      package->addSource(file.string());
    }
  }
  return package;
}

void birch::Driver::readFiles(const std::string& key) {
  auto tarName = tar(packageName);
  for (auto pattern : metaContents[key]) {
    auto paths = glob(pattern);
    if (paths.empty()) {
      /* print warning if pattern does not contain a wildcard '*' */
      if (pattern.find('*') == std::string::npos) {
        warn("no file matching '" + pattern + "' in build configuration.");
      }
    } else {
      for (auto path : paths) {
        if (std::regex_search(path.string(), std::regex("\\s",
            std::regex_constants::ECMAScript))) {
          throw DriverException(std::string("file name ") + path.string() +
            " in build configuration contains whitespace, which is not supported.");
        }
        if (!path.stem().compare(tarName) == 0) {
          auto inserted = allFiles.insert(path);
          if (!inserted.second) {
            warn(path.string() + " repeated in build configuration.");
          }
          metaFiles[key].push_back(path);
        }
      }
    }
  }
}
