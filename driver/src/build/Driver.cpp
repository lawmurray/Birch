/**
 * @file
 */
#include "src/build/Driver.hpp"

#include "src/build/MetaParser.hpp"
#include "src/build/Compiler.hpp"
#include "src/build/misc.hpp"
#include "src/generate/MarkdownGenerator.hpp"
#include "src/primitive/encode.hpp"
#include "src/exception/DriverException.hpp"

birch::Driver::Driver(int argc, char** argv) :
    packageName("Untitled"),
    packageVersion("unversioned"),
    unit("dir"),
    jobs(std::thread::hardware_concurrency()),
    debug(true),
    test(false),
    release(false),
    staticLib(false),
    sharedLib(true),
    openmp(true),
    warnings(true),
    notes(false),
    verbose(true),
    newBootstrap(false),
    newConfigure(false),
    newMake(false) {
  /* environment */
  char* BIRCH_MODE = getenv("BIRCH_MODE");
  char* BIRCH_PREFIX = getenv("BIRCH_PREFIX");
  char* BIRCH_SHARE_PATH = getenv("BIRCH_SHARE_PATH");
  char* BIRCH_INCLUDE_PATH = getenv("BIRCH_INCLUDE_PATH");
  char* BIRCH_LIBRARY_PATH = getenv("BIRCH_LIBRARY_PATH");
  std::string input;

  /* mode */
  if (BIRCH_MODE) {
    if (strcmp(BIRCH_MODE, "debug") == 0) {
      debug = true;
      test = false;
      release = false;
    } else if (strcmp(BIRCH_MODE, "test") == 0) {
      debug = false;
      test = true;
      release = false;
    } else if (strcmp(BIRCH_MODE, "release") == 0) {
      debug = false;
      test = false;
      release = true;
    }
  }

  /* prefix */
  if (prefix.empty()) {
    #ifdef PREFIX
    prefix = STRINGIFY(PREFIX);
    #endif
    if (BIRCH_PREFIX) {
      prefix = BIRCH_PREFIX;
    }
  }

  /* share dirs */
  if (BIRCH_SHARE_PATH) {
    std::stringstream birch_share_path(BIRCH_SHARE_PATH);
    while (std::getline(birch_share_path, input, ':')) {
      shareDirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    shareDirs.push_back(fs::path(prefix) / "share" / "birch");
  }
#ifdef DATADIR
  shareDirs.push_back(fs::path(STRINGIFY(DATADIR)) / "birch");
#endif

  /* include dirs */
  if (BIRCH_INCLUDE_PATH) {
    std::stringstream birch_include_path(BIRCH_INCLUDE_PATH);
    while (std::getline(birch_include_path, input, ':')) {
      includeDirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    includeDirs.push_back(fs::path(prefix) / "include");
  }
#ifdef INCLUDEDIR
  includeDirs.push_back(STRINGIFY(INCLUDEDIR));
#endif
  includeDirs.push_back(fs::path("/") / "usr" / "local" / "include");
  includeDirs.push_back(fs::path("/") / "usr" / "include");

  /* lib dirs */
  fs::path local = fs::path(".libs");
  if (fs::exists(local)) {
    libDirs.push_back(local);
  }
  if (BIRCH_LIBRARY_PATH) {
    std::stringstream birch_library_path(BIRCH_LIBRARY_PATH);
    while (std::getline(birch_library_path, input, ':')) {
      libDirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    if (fs::exists(fs::path(prefix) / "lib64")) {
      libDirs.push_back(fs::path(prefix) / "lib64");
    }
    if (fs::exists(fs::path(prefix) / "lib")) {
      libDirs.push_back(fs::path(prefix) / "lib");
    }
  }
#ifdef LIBDIR
  libDirs.push_back(STRINGIFY(LIBDIR));
#endif

  /* command-line options */
  enum {
    PACKAGE_ARG = 256,
    PREFIX_ARG,
    ARCH_ARG,
    MODE_ARG,
    UNIT_ARG,
    ENABLE_DEBUG_ARG,
    DISABLE_DEBUG_ARG,
    ENABLE_TEST_ARG,
    DISABLE_TEST_ARG,
    ENABLE_RELEASE_ARG,
    DISABLE_RELEASE_ARG,
    ENABLE_STATIC_ARG,
    DISABLE_STATIC_ARG,
    ENABLE_SHARED_ARG,
    DISABLE_SHARED_ARG,
    ENABLE_OPENMP_ARG,
    DISABLE_OPENMP_ARG,
    JOBS_ARG,
    ENABLE_WARNINGS_ARG,
    DISABLE_WARNINGS_ARG,
    ENABLE_NOTES_ARG,
    DISABLE_NOTES_ARG,
    ENABLE_VERBOSE_ARG,
    DISABLE_VERBOSE_ARG
  };

  int c, option_index;
  option long_options[] = {
      { "package", required_argument, 0, PACKAGE_ARG },
      { "prefix", required_argument, 0, PREFIX_ARG },
      { "arch", required_argument, 0, ARCH_ARG },
      { "unit", required_argument, 0, UNIT_ARG },
      { "jobs", required_argument, 0, JOBS_ARG },
      { "enable-debug", no_argument, 0, ENABLE_DEBUG_ARG },
      { "disable-debug", no_argument, 0, DISABLE_DEBUG_ARG },
      { "enable-test", no_argument, 0, ENABLE_TEST_ARG },
      { "disable-test", no_argument, 0, DISABLE_TEST_ARG },
      { "enable-release", no_argument, 0, ENABLE_RELEASE_ARG },
      { "disable-release", no_argument, 0, DISABLE_RELEASE_ARG },
      { "enable-static", no_argument, 0, ENABLE_STATIC_ARG },
      { "disable-static", no_argument, 0, DISABLE_STATIC_ARG },
      { "enable-shared", no_argument, 0, ENABLE_SHARED_ARG },
      { "disable-shared", no_argument, 0, DISABLE_SHARED_ARG },
      { "enable-openmp", no_argument, 0, ENABLE_OPENMP_ARG },
      { "disable-openmp", no_argument, 0, DISABLE_OPENMP_ARG },
      { "enable-warnings", no_argument, 0, ENABLE_WARNINGS_ARG },
      { "disable-warnings", no_argument, 0, DISABLE_WARNINGS_ARG },
      { "enable-notes", no_argument, 0, ENABLE_NOTES_ARG },
      { "disable-notes", no_argument, 0, DISABLE_NOTES_ARG },
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
    case JOBS_ARG:
      jobs = atoi(optarg);
      break;
    case ENABLE_DEBUG_ARG:
      debug = true;
      break;
    case DISABLE_DEBUG_ARG:
      debug = false;
      break;
    case ENABLE_TEST_ARG:
      test = true;
      break;
    case DISABLE_TEST_ARG:
      test = false;
      break;
    case ENABLE_RELEASE_ARG:
      release = true;
      break;
    case DISABLE_RELEASE_ARG:
      release = false;
      break;
    case ENABLE_STATIC_ARG:
      staticLib = true;
      break;
    case DISABLE_STATIC_ARG:
      staticLib = false;
      break;
    case ENABLE_SHARED_ARG:
      sharedLib = true;
      break;
    case DISABLE_SHARED_ARG:
      sharedLib = false;
      break;
    case ENABLE_OPENMP_ARG:
      openmp = true;
      break;
    case DISABLE_OPENMP_ARG:
      openmp = false;
      break;
    case ENABLE_WARNINGS_ARG:
      warnings = true;
      break;
    case DISABLE_WARNINGS_ARG:
      warnings = false;
      break;
    case ENABLE_NOTES_ARG:
      notes = true;
      break;
    case DISABLE_NOTES_ARG:
      notes = false;
      break;
    case ENABLE_VERBOSE_ARG:
      verbose = true;
      break;
    case DISABLE_VERBOSE_ARG:
      verbose = false;
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
}

void birch::Driver::run(const std::string& prog,
    const std::vector<char*>& xargv) {
  /* get package information */
  try {
    /* load the package meta information, if indeed there is any, otherwise
     * this will throw an exception which is caught below */
    meta();
  } catch (DriverException) {
    // probably not running in a package directory, but can use installed
    // libraries instead
  }

  /* name of the shared library file we expect to find */
  auto name = "lib" + tar(packageName);
  if (release) {
    // no suffix
  } else if (test) {
    name += "-test";
  } else if (debug) {
    name += "-debug";
  }
  fs::path so = name;
  #ifdef __APPLE__
  so.replace_extension(".dylib");
  #else
  so.replace_extension(".so");
  #endif

  /* dynamically load possible programs */
  typedef int prog_t(int argc, char** argv);
  void* handle;
  void* addr;
  char* msg;
  prog_t* fcn;

  auto path = find(libDirs, so);
  handle = dlopen(path.c_str(), RTLD_NOW);
  msg = dlerror();
  if (handle == NULL) {
    std::stringstream buf;
    buf << msg << '.';
    throw DriverException(buf.str());
  } else {
    addr = dlsym(handle, prog.c_str());
    msg = dlerror();
    if (msg != NULL) {
      std::stringstream buf;
      buf << "Could not find program " << prog << " in " << so.string()
          << '.';
      throw DriverException(buf.str());
    } else {
      auto argv = largv;
      argv.insert(argv.end(), xargv.begin(), xargv.end());
      fcn = reinterpret_cast<prog_t*>(addr);
      int ret = fcn(argv.size(), argv.data());
      if (ret != 0) {
        std::stringstream buf;
        buf << "Program " << prog << " exited with code " << ret << '.';
        throw DriverException(buf.str());
      }
    }
    dlclose(handle);
  }
}

void birch::Driver::bootstrap() {
  meta();
  setup();
  transpile();

  if (newBootstrap || newConfigure || newMake || !fs::exists("configure")) {
    std::stringstream cmd;
    cmd << (fs::path(".") / "bootstrap");
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > bootstrap.log 2>&1";
    }

    int status = std::system(cmd.str().c_str());
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::stringstream buf;
      buf << "bootstrap failed";
      if (!verbose) {
        buf << ", see bootstrap.log for details";
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
    if (release) {
      options << " --enable-release";
    } else {
      options << " --disable-release";
    }
    if (debug) {
      options << " --enable-debug";
    } else {
      options << " --disable-debug";
    }
    if (test) {
      options << " --enable-test";
    } else {
      options << " --disable-test";
    }
    if (staticLib) {
      options << " --enable-static";
    } else {
      options << " --disable-static";
    }
    if (sharedLib) {
      options << " --enable-shared";
    } else {
      options << " --disable-shared";
    }
    if (openmp) {
      #ifdef __APPLE__
      /* the system compiler on Apple requires different options for
        * OpenMP; disable the configure check and customize these */
      options << " --disable-openmp";
      cppflags << " -Xpreprocessor -fopenmp";
      #else
      options << " --enable-openmp";
      #endif
    } else {
      options << " --disable-openmp";
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
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > configure.log 2>&1";
    }

    int status = std::system(cmd.str().c_str());
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::stringstream buf;
      buf << "configure failed";
      if (!verbose) {
        buf << ", see configure.log and config.log for details";
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
  ldconfig();
}

void birch::Driver::uninstall() {
  configure();
  target("uninstall");
  ldconfig();
}

void birch::Driver::dist() {
  meta();

  /* determine archive name, format 'name-version' */
  auto archive = tar(packageName) + "-" + packageVersion;

  /* archiving command */
  std::stringstream cmd;
  #ifdef __APPLE__
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
  if (verbose) {
    std::cerr << cmd.str() << std::endl;
  } else {
    cmd << " > tar.log 2>&1";
  }

  /* run command */
  int status = std::system(cmd.str().c_str());
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::stringstream buf;
    buf << "tar failed";
    if (!verbose) {
      buf << ", see tar.log for details";
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
  fs::remove("depcomp");
  fs::remove("install-sh");
  fs::remove("libtool");
  fs::remove("ltmain.sh");
  fs::remove("Makefile");
  fs::remove("Makefile.am");
  fs::remove("Makefile.in");
  fs::remove("missing");
  fs::remove("lib" + tarName + "-debug.la");
  fs::remove("lib" + tarName + "-test.la");
  fs::remove("lib" + tarName + ".la");
  fs::remove(tarName + ".birch");
  fs::remove(tarName + ".hpp");

  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    fs::path source = tarName;
    source.replace_extension(".cpp");
    fs::remove(source);
    source.replace_extension(".lo");

    fs::path object;
    object = source.parent_path() / ("lib" + canonicalName + "_debug_la-" + source.filename().string());
    fs::remove(object);
    object = source.parent_path() / ("lib" + canonicalName + "_test_la-" + source.filename().string());
    fs::remove(object);
    object = source.parent_path() / ("lib" + canonicalName + "_la-" + source.filename().string());
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
        source.replace_extension(".lo");

        fs::path object;
        object = source.parent_path() / ("lib" + canonicalName + "_debug_la-" + source.filename().string());
        fs::remove(object);
        object = source.parent_path() / ("lib" + canonicalName + "_test_la-" + source.filename().string());
        fs::remove(object);
        object = source.parent_path() / ("lib" + canonicalName + "_la-" + source.filename().string());
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
        object = source.parent_path() / ("lib" + canonicalName + "_debug_la-" + source.filename().string());
        fs::remove(object);
        object = source.parent_path() / ("lib" + canonicalName + "_test_la-" + source.filename().string());
        fs::remove(object);
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
      iter.no_push();
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
  Package* package = createPackage(false);

  /* parse all files */
  Compiler compiler(package, unit);
  compiler.parse(false);

  /* output everything into single file */
  fs::ofstream docsStream("DOCS.md");
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
  fs::create_directories(docs / "unary_operators");
  fs::create_directories(docs / "binary_operators");
  fs::create_directories(docs / "classes");

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

      /* among first-level headers, only variables and types have their own
       * page, rather than being further split into a page per item */
      if (h1 == "Variables" || h1 == "Types") {
        std::string dir = h1;
        boost::to_lower(dir);
        file = fs::path(dir) / "index.md";
        if (docsStream.is_open()) {
          docsStream.close();
        }
        docsStream.open(docs / file);
        docsStream << "# " << h1 << "\n\n";
      }
      boost::to_lower(h1);
      boost::replace_all(h1, " ", "_");
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
      std::cout << "Basic options:" << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-debug / --disable-debug (default enabled):" << std::endl;
      std::cout << "  Enable/disable debug mode build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-test / --disable-test (default disabled):" << std::endl;
      std::cout << "  Enable/disable test mode build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-release / --disable-release (default enabled):" << std::endl;
      std::cout << "  Enable/disable release mode build." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-warnings / --disable-warnings (default enabled):" << std::endl;
      std::cout << "  Enable/disable compiler warnings." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-notes / --disable-notes (default disabled):" << std::endl;
      std::cout << "  Enable/disable compiler notes." << std::endl;
      std::cout << std::endl;
      std::cout << "  --enable-verbose / --disable-verbose (default enabled):" << std::endl;
      std::cout << "  Show all compiler output." << std::endl;
      std::cout << std::endl;
      std::cout << "  --prefix (default imputed): Installation prefix. Defaults to the same prefix" << std::endl;
      std::cout << "  used when installing the birch driver program." << std::endl;
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

  /* check manifest files */
  readFiles("manifest.header");
  readFiles("manifest.source");
  readFiles("manifest.data");
  readFiles("manifest.other");
}

void birch::Driver::setup() {
  auto tarName = tar(packageName);
  auto canonicalName = canonical(packageName);

  /* copy build files */
  newBootstrap = copy_if_newer(find(shareDirs, "bootstrap"), "bootstrap");
  fs::permissions("bootstrap", fs::add_perms|fs::owner_exe);

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
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_VERSION", packageVersion);
  boost::replace_all(contents, "PACKAGE_TARNAME", tarName);
  boost::replace_all(contents, "PACKAGE_CANONICAL_NAME", canonicalName);
  std::stringstream configureStream;
  configureStream << contents << "\n\n";

  /* required headers */
  for (auto value : metaContents["require.header"]) {
    configureStream << "AC_CHECK_HEADERS([" << value << "], [], [AC_MSG_ERROR([required header not found.])], [-])\n";
  }
  for (auto value : metaContents["require.package"]) {
    auto tarName = tar(value);
    configureStream << "AC_CHECK_HEADERS([" << tarName << ".hpp], [], [AC_MSG_ERROR([required header not found.])], [-])\n";
  }

  /* required libraries */
  for (auto value : metaContents["require.library"]) {
    configureStream << "AC_CHECK_LIB([" << value << "], [main], [], [AC_MSG_ERROR([required library not found.])])\n";
  }
  for (auto value : metaContents["require.package"]) {
    auto tarName = tar(value);
    configureStream << "if $debug; then\n";
    configureStream << "  AC_CHECK_LIB([" << tarName << "-debug], [main], [DEBUG_LIBS=\"$DEBUG_LIBS -l" << tarName << "-debug\"], [AC_MSG_ERROR([required library not found.])], [$DEBUG_LIBS])\n";
    configureStream << "fi\n";
    configureStream << "if $test; then\n";
    configureStream << "  AC_CHECK_LIB([" << tarName << "-test], [main], [TEST_LIBS=\"$TEST_LIBS -l" << tarName << "-test\"], [AC_MSG_ERROR([required library not found.])], [$TEST_LIBS])\n";
    configureStream << "fi\n";
    configureStream << "if $release; then\n";
    configureStream << "  AC_CHECK_LIB([" << tarName << "], [main], [RELEASE_LIBS=\"$RELEASE_LIBS -l" << tarName << "\"], [AC_MSG_ERROR([required library not found.])], [$RELEASE_LIBS])\n";
    configureStream << "fi\n";
  }

  /* required programs */
  for (auto value : metaContents["require.program"]) {
    configureStream << "  AC_PATH_PROG([PROG], [" << value << "], [])\n";
    configureStream << "  if test \"$PROG\" = \"\"; then\n";
    configureStream << "    AC_MSG_ERROR([required program not found.])\n";
    configureStream << "  fi\n";
  }

  /* footer */
  configureStream << "AC_SUBST([DEBUG_LIBS])\n";
  configureStream << "AC_SUBST([TEST_LIBS])\n";
  configureStream << "AC_SUBST([RELEASE_LIBS])\n";
  configureStream << "\n";
  configureStream << "AC_CONFIG_FILES([Makefile])\n";
  configureStream << "AC_OUTPUT\n";

  newConfigure = write_all_if_different("configure.ac",
      configureStream.str());

  /* Makefile.am */
  contents = read_all(find(shareDirs, "Makefile.am"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_VERSION", packageVersion);
  boost::replace_all(contents, "PACKAGE_TARNAME", tarName);
  boost::replace_all(contents, "PACKAGE_CANONICAL_NAME", canonicalName);

  std::stringstream makeStream;
  makeStream << contents << "\n\n";
  makeStream << "COMMON_SOURCES =";
  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    auto source = fs::path(tarName);
    source.replace_extension(".cpp");
    makeStream << " \\\n  " << source.string() << ".cpp";
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
  makeStream << '\n';

  /* headers to install and distribute */
  makeStream << "include_HEADERS =";
  auto header = fs::path(tarName);
  header.replace_extension(".hpp");
  makeStream << " \\\n  " << header.string();
  header.replace_extension(".birch");
  makeStream << " \\\n  " << header.string();
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
  Compiler compiler(createPackage(true), unit);
  compiler.parse(true);
  compiler.resolve();
  compiler.gen();
}

void birch::Driver::target(const std::string& cmd) {
  /* command */
  std::stringstream buf;
  if (arch == "js" || arch == "wasm") {
    buf << "emmake";
  }
  buf << "make -s LIBTOOLFLAGS=--silent";

  /* concurrency */
  if (jobs > 1) {
    buf << " -j " << jobs;
  }

  /* target */
  buf << ' ' << cmd << " 2>&1";  // stderr to stdout so pipe has both

  if (verbose) {
    std::cerr << buf.str() << std::endl;
  }

  /* create a filter for the output of make to make it digestable; while we
   * can use pipes to grep and sed within the shell command, this is less
   * portable, and means the command always returns success, even on fail
   * (consider pipefail); instead we use popen instead of system, and process
   * the output with regexes */
  std::string type1 = "[a-zA-Z0-9_\\[\\]\\?,]+";
  std::string type2 = type1 + "(?:<" + type1 + " *>)?(?:\\[[_,]+\\])?";
  std::string type3 = type1 + "(?:<" + type2 + " *>)?(?:\\[[_,]+\\])?";
  std::string type4 = type1 + "(?:<" + type3 + " *>)?(?:\\[[_,]+\\])?";
  std::string type = type4;

  std::regex rxWarnings("warning:");
  std::regex rxNotes("note:");
  std::regex rxSkipLine("In file included from|In member function|^\\s*from|std::enable_if|At global scope:");
  std::regex rxNamespace("birch::type::|birch::|libbirch::");
  std::regex rxCxxWords("virtual *");
  std::regex rxWith("\\[with.*?\\]");
  std::regex rxReal("double");
  std::regex rxInteger("long int");
  std::regex rxLazy("Lazy<Shared<(" + type + ") *> *> *");
  std::regex rxOptional("Optional<(" + type + ") *> *");
  std::regex rxVector("Array<(" + type + "), *Shape<Dimension<(?:0, *0)?>, *EmptyShape *> *> *");
  std::regex rxVector2("DefaultArray<(" + type + "), *1> *");
  std::regex rxMatrix("Array<(" + type + "), *Shape<Dimension<(?:0, *0)?>, Shape<Dimension<(?:0, *0)?>, *EmptyShape *> *> *> *");
  std::regex rxMatrix2("DefaultArray<(" + type + "), *2> *");
  std::regex rxConstRef("(?:const *)?(" + type + ") *&");
  std::regex rxLLT("Eigen::LLT<Eigen::Matrix<double, *-1, *-1, *1, *-1, *-1> *> *");
  std::regex rxThis("\\(\\(" + type + "\\*\\)this\\)->" + type + "::this_\\(\\)->" + type + "::");
  std::regex rxAka("\\{aka *‘?(class |const )?" + type + "’?\\} *");
  std::regex rxDeref("operator->");
  std::regex rxDerefExpr("‘->’");
  std::regex rxAssign("operator=");
  std::regex rxAssignExpr("‘=’");
  std::regex rxHandler("(?:, *)?Handler *(?:\\* *& *handler_)?");
  std::regex rxHandler2("(?:, *)?\\(\\* *& *handler_\\)");
  std::regex rxTooFewArguments("(?:invalid initialization of reference of type ‘" + type4 + "’ from expression of type ‘Handler’|cannot convert ‘Handler’ to ‘" + type4 + "’)");

  std::ofstream log;
  if (!verbose) {
    log.open("make.log", std::ios_base::out);
  }
  FILE* pipe = popen(buf.str().c_str(), "r");
  if (pipe) {
    char* line = NULL;
    size_t n = 0;
    while (getline(&line, &n, pipe) > 0) {
      std::string str(line);
      free(line);
      line = NULL;
      n = 0;

      if ((warnings || !std::regex_search(str, rxWarnings)) &&
          (notes || !std::regex_search(str, rxNotes)) &&
          !std::regex_search(str, rxSkipLine)) {
        /* strip namespace and class qualifiers */
        str = std::regex_replace(str, rxNamespace, "");

        /* strip "with" type hints */
        str = std::regex_replace(str, rxWith, "");

        /* strip some C++ words */
        str = std::regex_replace(str, rxCxxWords, "");

        /* convert back some types */
        str = std::regex_replace(str, rxReal, "Real");
        str = std::regex_replace(str, rxInteger, "Integer");

        /* replace some types; repeat some of these patterns a few times
         * as a hacky way of handling recursion */
        for (auto i = 0; i < 3; ++i) {
          str = std::regex_replace(str, rxVector, "$1[_]");
          str = std::regex_replace(str, rxVector2, "$1[_]");
          str = std::regex_replace(str, rxMatrix, "$1[_,_]");
          str = std::regex_replace(str, rxMatrix2, "$1[_,_]");
          str = std::regex_replace(str, rxLLT, "LLT");
          str = std::regex_replace(str, rxLazy, "$1");
          str = std::regex_replace(str, rxOptional, "$1?");
          str = std::regex_replace(str, rxConstRef, "$1");
        }
        str = std::regex_replace(str, rxAka, "");
        str = std::regex_replace(str, rxThis, "");

        /* replace some operators */
        str = std::regex_replace(str, rxDeref, ".");
        str = std::regex_replace(str, rxDerefExpr, "‘.’");
        str = std::regex_replace(str, rxAssign, "<-");
        str = std::regex_replace(str, rxAssignExpr, "‘<-’");

        /* strip suggestions that reveal internal workings */
        str = std::regex_replace(str, rxHandler, "");
        str = std::regex_replace(str, rxHandler2, "");
        str = std::regex_replace(str, rxTooFewArguments, "too few arguments to function call");

        if (verbose) {
         std::cerr << str;
        } else {
         log << str;
        }
      }
    }
    if (!verbose) {
      log.close();
    }
    int status = pclose(pipe);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::stringstream buf;
      buf << "make failed";
      if (!verbose) {
        buf << ", see make.log for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void birch::Driver::ldconfig() {
  #ifndef __APPLE__
  auto euid = geteuid();
  if (euid == 0) {
    [[maybe_unused]] int result = std::system("ldconfig");
  }
  #endif
}

birch::Package* birch::Driver::createPackage(bool includeRequires) {
  Package* package = new Package(packageName);
  if (includeRequires) {
    for (auto value : metaContents["require.package"]) {
      package->addPackage(value);

      /* add *.birch dependency */
      fs::path header = tar(value);
      header.replace_extension(".birch");
      package->addHeader(find(includeDirs, header).string());
    }
  }
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".birch") == 0) {
      package->addSource(file.string());
    }
  }
  return package;
}

void birch::Driver::readFiles(const std::string& key) {
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
        auto inserted = allFiles.insert(path);
        if (!inserted.second) {
          warn(path.string() + " repeated in build configuration.");
        }
        metaFiles[key].push_back(path);
      }
    }
  }
}
