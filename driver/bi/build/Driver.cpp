/**
 * @file
 */
#include "Driver.hpp"

#include "bi/build/Compiler.hpp"
#include "bi/build/misc.hpp"
#include "bi/io/md_ostream.hpp"
#include "bi/primitive/encode.hpp"
#include "bi/exception/DriverException.hpp"

bi::Driver::Driver(int argc, char** argv) :
    /* keep paths relative, or at least call configure with a
     * relative path from the build directory to the work directory,
     * otherwise a work directory containing spaces causes problems */
    packageName("Untitled"),
    packageVersion("unversioned"),
    arch("native"),
    mode("debug"),
    unit("dir"),
    jobs(std::thread::hardware_concurrency()),
    staticLib(false),
    sharedLib(true),
    openmp(true),
    warnings(true),
    notes(true),
    verbose(true),
    newBootstrap(false),
    newConfigure(false),
    newMake(false) {
  enum {
    PACKAGE_ARG = 256,
    PREFIX_ARG,
    ARCH_ARG,
    MODE_ARG,
    UNIT_ARG,
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
      { "mode", required_argument, 0, MODE_ARG },
      { "unit", required_argument, 0, UNIT_ARG },
      { "jobs", required_argument, 0, JOBS_ARG },
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
    case MODE_ARG:
      mode = optarg;
      break;
    case UNIT_ARG:
      unit = optarg;
      break;
    case JOBS_ARG:
      jobs = atoi(optarg);
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
  if (arch != "native" && arch != "js" && arch != "wasm") {
    throw DriverException("--arch must be native, js, or wasm.");
  }
  if (mode != "debug" && mode != "test" && mode != "release") {
    throw DriverException("--mode must be latest, debug, test, or release.");
  }
  if (unit != "unity" && unit != "dir" && unit != "file") {
    throw DriverException("--unit must be unity, dir, or file.");
  }

  /* environment variables */
  char* BIRCH_PREFIX = getenv("BIRCH_PREFIX");
  char* BIRCH_SHARE_PATH = getenv("BIRCH_SHARE_PATH");
  char* BIRCH_INCLUDE_PATH = getenv("BIRCH_INCLUDE_PATH");
  char* BIRCH_LIBRARY_PATH = getenv("BIRCH_LIBRARY_PATH");
  std::string input;

  /* install prefix */
  if (prefix.empty() && BIRCH_PREFIX) {
    prefix = BIRCH_PREFIX;
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
  includeDirs.push_back(fs::absolute("src"));
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

  /* lib dirs */
  fs::path local = fs::path("build") / "latest" / ".libs";
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
}

void bi::Driver::run(const std::string& prog,
    const std::vector<char*>& xargv) {
  /* get package information */
  try {
    /* load the package meta information, if indeed there is any, otherwise
     * this will throw an exception which is caught below */
    meta();

    /* when running in a package directory we'd like to use the latest build,
     * regardless of what mode was given in the call, so update the mode if
     * we find a latest build */
    auto symlink = fs::path("build") / "latest";
    if (fs::exists(symlink)) {
      auto build = fs::read_symlink(symlink).string();
      if (build.find("debug_") == 0) {
        mode = "debug";
      } else if (build.find("test_") == 0) {
        mode = "test";
      } else if (build.find("release_") == 0) {
        mode = "release";
      }
    }
  } catch (DriverException) {
    // probably not running in a package directory, but can use installed
    // libraries instead
  }

  /* name of the shared library file we expect to find */
  auto name = "lib" + tarname(packageName);
  if (mode != "release") {
    name += '_' + mode;
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

void bi::Driver::build() {
  meta();
  setup();
  transpile();
  bootstrap();
  configure();
  target();
  if (arch == "js" || arch == "wasm") {
    target("birch.html");
  }
}

void bi::Driver::install() {
  meta();
  target("install");
  ldconfig();
}

void bi::Driver::uninstall() {
  meta();
  target("uninstall");
  ldconfig();
}

void bi::Driver::dist() {
  meta();

  /* determine archive name, format 'name-version' */
  auto archive = tarname(packageName) + "-" + packageVersion;

  /* archiving command */
  std::stringstream cmd;
  cmd << "tar czf " << archive << ".tar.gz ";
  cmd << "--transform=\"s/^/" << archive << "\\//\"";
  for (auto key : {
      "manifest.header",
      "manifest.source",
      "manifest.data",
      "manifest.other"}) {
    for (auto file : metaFiles[key]) {
      cmd << ' ' << file;
    }
  }

  /* run command */
  int ret = std::system(cmd.str().c_str());
  if (ret == -1) {
    throw DriverException(explain(cmd.str()));
  }
}

void bi::Driver::clean() {
  meta();

  fs::remove_all("build");
  fs::remove_all("autom4te.cache");
  fs::remove_all("m4");
  fs::remove_all("src");
  fs::remove("aclocal.m4");
  fs::remove("bootstrap.log");
  fs::remove("bootstrap");
  fs::remove("compile");
  fs::remove("config.guess");
  fs::remove("config.sub");
  fs::remove("configure");
  fs::remove("configure.ac");
  fs::remove("depcomp");
  fs::remove("install-sh");
  fs::remove("ltmain.sh");
  fs::remove("Makefile.am");
  fs::remove("Makefile.in");
  fs::remove("missing");
}

void bi::Driver::init() {
  fs::create_directory("bi");
  fs::create_directory("config");
  fs::create_directory("input");
  fs::create_directory("output");
  copy_with_prompt(find(shareDirs, "gitignore"), ".gitignore");
  copy_with_prompt(find(shareDirs, "LICENSE"), "LICENSE");

  std::string contents;

  contents = read_all(find(shareDirs, "META.json"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream metaStream("META.json");
  if (metaStream.fail()) {
    std::stringstream buf;
    buf << "Could not open META.json for writing.";
    throw DriverException(buf.str());
  }
  metaStream << contents;

  contents = read_all(find(shareDirs, "README.md"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream readmeStream("README.md");
  if (readmeStream.fail()) {
    std::stringstream buf;
    buf << "Could not open README.md for writing.";
    throw DriverException(buf.str());
  }
  readmeStream << contents;
}

void bi::Driver::check() {
  /* read META.json */
  if (!fs::exists("META.json")) {
    warn("no META.json file.");
  } else {
    meta();
  }

  /* check LICENSE */
  if (!fs::exists("LICENSE")) {
    warn("no LICENSE file; create a LICENSE file containing the "
        "distribution license (e.g. GPL or BSD) of the package.");
  } else if (allFiles.find("LICENSE") == allFiles.end()) {
    warn("LICENSE file is not listed in META.json file.");
  }

  /* check README.md */
  if (!fs::exists("README.md")) {
    warn("no README.md file; create a README.md file documenting the "
        "package in Markdown format.");
  } else if (allFiles.find("README.md") == allFiles.end()) {
    warn("README.md file is not listed in META.json file.");
  }

  /* check for files that might be missing from META.json */
  std::unordered_set<std::string> interesting, exclude;

  interesting.insert(".bi");
  interesting.insert(".sh");
  interesting.insert(".json");
  interesting.insert(".yml");
  interesting.insert(".cpp");
  interesting.insert(".hpp");

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
        warn(
            std::string("is ") + path.string()
                + " missing from META.json file?");
      }
    }
    ++iter;
  }
}

void bi::Driver::docs() {
  meta();
  Package* package = createPackage(false);

  /* parse all files */
  Compiler compiler(package, mode, unit);
  compiler.parse(false);

  /* output everything into single file */
  fs::ofstream docsStream("DOCS.md");
  if (docsStream.fail()) {
    std::stringstream buf;
    buf << "Could not open DOCS.md for writing.";
    throw DriverException(buf.str());
  }
  md_ostream output(docsStream);
  output << package;
  docsStream.close();

  /* split that file into multiple files for mkdocs */
  fs::ofstream mkdocsStream("mkdocs.yml");
  if (mkdocsStream.fail()) {
    std::stringstream buf;
    buf << "Could not open mkdocs.yml for writing.";
    throw DriverException(buf.str());
  }
  mkdocsStream << "site_name: '" << packageName << "'\n";
  mkdocsStream << "theme:\n";
  mkdocsStream << "  name: 'material'\n";
  mkdocsStream << "markdown_extensions:\n";
  mkdocsStream << "  - admonition\n";
  mkdocsStream << "  - footnotes\n";
  mkdocsStream << "  - pymdownx.arithmatex\n";
  mkdocsStream << "  - pymdownx.superfences:\n";
  mkdocsStream << "      custom_fences:\n";
  mkdocsStream << "        - name: mermaid\n";
  mkdocsStream << "          class: mermaid\n";
  mkdocsStream << "          format: !!python/name:pymdownx.superfences.fence_div_format\n";
  mkdocsStream << "extra_css:\n";
  mkdocsStream << "  - 'https://unpkg.com/mermaid@8.5.1/dist/mermaid.css'\n";
  mkdocsStream << "extra_javascript:\n";
  mkdocsStream << "  - 'https://unpkg.com/mermaid@8.5.1/dist/mermaid.min.js'\n";
  mkdocsStream << "  - 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'\n";
  mkdocsStream << "nav:\n";

  fs::path docs("docs"), file;
  fs::create_directories(docs);
  fs::create_directories(docs / "types");
  fs::create_directories(docs / "variables");
  fs::create_directories(docs / "programs");
  fs::create_directories(docs / "functions");
  fs::create_directories(docs / "fibers");
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
  mkdocsStream << "  - index.md\n";

  std::string str = read_all("DOCS.md");
  std::regex reg("(?:^|\r?\n)(##?) (.*?)(?=\r?\n|$)",
      std::regex_constants::ECMAScript);
  std::smatch match;
  std::string str1 = str, h1, h2;
  while (std::regex_search(str1, match, reg)) {
    if (docsStream.is_open()) {
      docsStream << match.prefix();
    }
    if (match.str(1) == "#") {
      /* first level header */
      h1 = match.str(2);
      mkdocsStream << "  - '" << h1 << "': ";

      /* among first-level headers, only variables and types have their own
       * page, rather than being further split into a page per item */
      if (h1 == "Variables" || h1 == "Types") {
        std::string dir = h1;
        boost::to_lower(dir);
        file = fs::path(dir) / "index.md";
        mkdocsStream << file.string();
        if (docsStream.is_open()) {
          docsStream.close();
        }
        docsStream.open(docs / file);
        docsStream << "# " << h1 << "\n\n";
      }
      mkdocsStream << '\n';
      boost::to_lower(h1);
      boost::replace_all(h1, " ", "_");
    } else {
      /* second level header */
      h2 = match.str(2);
      mkdocsStream << "    - '" << h2 << "': ";
      file = fs::path(nice(h1)) / (nice(h2) + ".md");
      mkdocsStream << file.string() << "\n";
      if (docsStream.is_open()) {
        docsStream.close();
      }
      docsStream.open(docs / file);
    }
    str1 = match.suffix();
  }
  if (docsStream.is_open()) {
    docsStream << str1;
    docsStream.close();
  }
}

void bi::Driver::help() {
  std::cout << std::endl;
  if (largv.size() >= 2) {
    std::string command = largv.at(1);
    if (command.compare("init") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch init [options]" << std::endl;
      std::cout << std::endl;
      std::cout << "Initialise the working directory for a new package." << std::endl;
      std::cout << std::endl;
      std::cout << "  --package (default Untitled): Name of the package." << std::endl;
    } else if (command.compare("check") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch check" << std::endl;
      std::cout << std::endl;
      std::cout << "Check the file structure of the package for possible issues. This makes no" << std::endl;
      std::cout << "modifications to the package, but will output warnings for possible issues such" << std::endl;
      std::cout << "as:" << std::endl;
      std::cout << std::endl;
      std::cout << "  * files listed in META.json that do not exist," << std::endl;
      std::cout << "  * files of recognisable types that exist but are not listed in META.json, and" << std::endl;
      std::cout << "  * standard meta files that do not exist." << std::endl;
    } else if (command.compare("build") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch build [options]" << std::endl;
      std::cout << std::endl;
      std::cout << "Build the package." << std::endl;
      std::cout << std::endl;
      std::cout << "Basic options:" << std::endl;
      std::cout << std::endl;
      std::cout << "  --mode={debug|test|release} (default debug):" << std::endl;
      std::cout << "  Build for debugging (no optimizations, all assertion checks), testing" << std::endl;
      std::cout << "  (debugging plus code coverage) or release (all optimizations, no assertion" << std::endl;
      std::cout << "  checks)." << std::endl;
      std::cout << std::endl;
      std::cout << "  --jobs (default imputed):" << std::endl;
      std::cout << "  Number of jobs for a parallel build. By default, a reasonable value is" << std::endl;
      std::cout << "  determined from the environment." << std::endl;
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
      std::cout << "Documentation for the advanced options can be found at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://birch-lang.org/documentation/driver/commands/build/" << std::endl;
    } else if (command.compare("install") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch install [options]" << std::endl;
      std::cout << std::endl;
      std::cout << "Install the package after building. Accepts the same options as birch build," << std::endl;
      std::cout << "and indeed should be used with the same options as the preceding build." << std::endl;
      std::cout << std::endl;
      std::cout << "This installs all header, library and data files needed by the package into" << std::endl;
      std::cout << "the directory specified by --prefix (or the default if this was not specified)." << std::endl;
    } else if (command.compare("uninstall") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch uninstall" << std::endl;
      std::cout << std::endl;
      std::cout << "Uninstall the package. This uninstalls all header, library and data files from" << std::endl;
      std::cout << "the directory specified by --prefix (or the system default if this was not" << std::endl;
      std::cout << "specified)." << std::endl;
    } else if (command.compare("dist") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch dist" << std::endl;
      std::cout << std::endl;
      std::cout << "Build a distributable archive for the package." << std::endl;
      std::cout << std::endl;
      std::cout << "More information can be found at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://birch-lang.org/documentation/driver/commands/dist/" << std::endl;
    } else if (command.compare("docs") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch docs" << std::endl;
      std::cout << std::endl;
      std::cout << "Build the reference documentation for the package. This creates a Markdown file" << std::endl;
      std::cout << "DOCS.md in the current working directory." << std::endl;
      std::cout << std::endl;
      std::cout << "It will be overwritten if it already exists, and may be readily converted to" << std::endl;
      std::cout << "other formats using a utility such as pandoc." << std::endl;
      std::cout << std::endl;
      std::cout << "More information can be found at:" << std::endl;
      std::cout << std::endl;
      std::cout << "  https://birch-lang.org/documentation/driver/commands/docs/" << std::endl;
    } else if (command.compare("clean") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch clean" << std::endl;
      std::cout << std::endl;
      std::cout << "Clean the package directory of all build files." << std::endl;
    } else if (command.compare("help") == 0) {
      std::cout << "Usage:" << std::endl;
      std::cout << std::endl;
      std::cout << "  birch help [command]" << std::endl;
      std::cout << std::endl;
      std::cout << "Print the help message." << std::endl;
    } else {
      std::cout << "Command " << largv.at(1) << " is not a valid command."  << std::endl;
    }
  } else {
    std::cout << "Usage:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch <command> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Available commands:" << std::endl;
    std::cout << std::endl;
    std::cout << "  init          Initialize the working directory for a new package." << std::endl;
    std::cout << "  check         Check the file structure of the package for possible issues." << std::endl;
    std::cout << "  build         Build the package." << std::endl;
    std::cout << "  install       Install the package after building." << std::endl;
    std::cout << "  uninstall     Uninstall the package." << std::endl;
    std::cout << "  dist          Build a distributable archive for the package." << std::endl;
    std::cout << "  docs          Build the reference documentation for the package." << std::endl;
    std::cout << "  clean         Clean the package directory of all build files." << std::endl;
    std::cout << "  help          Print this help message." << std::endl;
    std::cout << std::endl;
    std::cout << "To print more detailed description of a command, including available options," << std::endl;
    std::cout << "use:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch help <command>" << std::endl;
    std::cout << std::endl;
    std::cout << "To call a program defined in the package use:" << std::endl;
    std::cout << std::endl;
    std::cout << "  birch <program name> [program options]" << std::endl;
    std::cout << std::endl;
    std::cout << "More information can be found at:" << std::endl;
    std::cout << std::endl;
    std::cout << "  https://birch-lang.org/" << std::endl;
  }
  std::cout << std::endl;
}

void bi::Driver::meta() {
  /* clear any previous read */
  metaFiles.clear();
  allFiles.clear();

  /* check for META.json */
  if (!fs::exists("META.json")) {
    throw DriverException("META.json does not exist.");
  }

  /* parse META.json */
  boost::property_tree::ptree meta;
  try {
    boost::property_tree::read_json("META.json", meta);
  } catch (boost::exception& e) {
    throw DriverException("syntax error in META.json.");
  }

  /* meta */
  if (auto name = meta.get_optional<std::string>("name")) {
    packageName = name.get();
  } else {
    throw DriverException(
        "META.json must provide a 'name' entry with the name of this package.");
  }
  if (auto description = meta.get_optional<std::string>("description")) {
    packageDescription = description.get();
  }
  if (auto version = meta.get_optional<std::string>("version")) {
    packageVersion = version.get();
  }

  /* external requirements */
  if (packageName != "Standard") {
    /* implicitly include the standard library, if this package is not,
     * itself, the standard library */
    metaFiles["require.package"].push_front("Standard");
  }
  readFiles(meta, "require.package", false);
  readFiles(meta, "require.header", false);
  readFiles(meta, "require.library", false);
  readFiles(meta, "require.program", false);

  /* manifest */
  readFiles(meta, "manifest.header", true);
  readFiles(meta, "manifest.source", true);
  readFiles(meta, "manifest.data", true);
  readFiles(meta, "manifest.other", true);
}

void bi::Driver::setup() {
  auto buildDir = fs::path("build") / suffix();
  auto internalName = tarname(packageName);

  /* create build directory */
  if (!fs::exists(buildDir)) {
    if (!fs::create_directories(buildDir)) {
      std::stringstream buf;
      buf << "could not create build directory " << buildDir << '.';
      throw DriverException(buf.str());
    }
  }

  /* update "latest" symlink to point to this build directory */
  auto symlink_dir = fs::path("build") / "latest";
  fs::remove(symlink_dir);
  fs::create_symlink(suffix(), symlink_dir);

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

  /* update configure.ac */
  std::string contents = read_all(find(shareDirs, "configure.ac"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_VERSION", packageVersion);
  boost::replace_all(contents, "PACKAGE_TARNAME", internalName);
  std::stringstream configureStream;
  configureStream << contents << "\n\n";

  /* required headers */
  if (!metaFiles["require.header"].empty()) {
    configureStream << "if test x$emscripten = xfalse; then\n";
  }
  for (auto file : metaFiles["require.header"]) {
    configureStream << "  AC_CHECK_HEADERS([" << file.string() << "], [], "
        << "[AC_MSG_ERROR([required header not found.])], [-])\n";
  }
  for (auto name : metaFiles["require.package"]) {
    auto internalName = tarname(name.string());
    configureStream << "  AC_CHECK_HEADERS([" << internalName << ".hpp], [], "
        << "[AC_MSG_ERROR([required header not found.])], [-])\n";
  }
  if (!metaFiles["require.header"].empty()) {
    configureStream << "fi\n";
  }

  /* required libraries */
  for (auto file : metaFiles["require.library"]) {
    configureStream << "  AC_CHECK_LIB([" << file.string() << "], [main], "
        << "[], [AC_MSG_ERROR([required library not found.])])\n";
  }
  for (auto name : metaFiles["require.package"]) {
    auto internalName = tarname(name.string());
    configureStream << "  AC_CHECK_LIB([" << internalName <<
        "$SUFFIX], [main], [], " <<
        "[AC_MSG_ERROR([required library not found.])])\n";
  }

  /* required programs */
  for (auto file : metaFiles["require.program"]) {
    configureStream << "  AC_PATH_PROG([PROG], [" << file.string() <<
        "], [])\n";
    configureStream << "  if test \"$PROG\" = \"\"; then\n";
    configureStream << "    AC_MSG_ERROR([required program not found.])\n";
    configureStream << "  fi\n";
  }

  /* footer */
  configureStream << "AC_CONFIG_FILES([Makefile])\n";
  configureStream << "AC_OUTPUT\n";

  newConfigure = write_all_if_different("configure.ac",
      configureStream.str());

  /* update Makefile.am */
  contents = read_all(find(shareDirs, "Makefile.am"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_TARNAME", internalName);

  std::stringstream makeStream;
  makeStream << contents << "\n\n";
  makeStream << "lib_LTLIBRARIES = lib" << internalName << "@SUFFIX@.la\n\n";

  /* sources derived from *.bi files */
  makeStream << "dist_lib" << internalName << "@SUFFIX@_la_SOURCES =";
  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    auto source = fs::path("src") / internalName;
    source.replace_extension(".cpp");
    makeStream << " \\\n  " << source.string() << ".cpp";
  } else if (unit == "file") {
    /* sources go into one *.cpp file for each *.bi file */
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".bi") == 0) {
        auto source = file;
        source.replace_extension(".cpp");
        makeStream << " \\\n  " << source.string();
      }
    }
  } else {
    /* sources go into one *.cpp file for each directory */
    std::unordered_set<std::string> sources;
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".bi") == 0) {
        auto source = fs::path("src") / file.parent_path() / internalName;
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
  auto header = fs::path("src") / internalName;
  header.replace_extension(".hpp");
  makeStream << " \\\n  " << header.string();
  header.replace_extension(".bih");
  makeStream << " \\\n  " << header.string();
  makeStream << '\n';

  /* data files to distribute */
  makeStream << "dist_pkgdata_DATA = ";
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

void bi::Driver::transpile() {
  Compiler compiler(createPackage(true), mode, unit);
  compiler.parse(true);
  compiler.resolve();
  compiler.gen();
}

void bi::Driver::bootstrap() {
  if (newBootstrap || newConfigure || newMake || !fs::exists("configure") ||
      !fs::exists("install-sh")) {
    std::stringstream cmd;
    cmd << (fs::path(".") / "bootstrap");
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > bootstrap.log 2>&1";
    }

    int ret = std::system(cmd.str().c_str());
    if (ret == -1) {
      if (verbose) {
        std::cerr << explain(cmd.str()) << std::endl;
      }
      throw DriverException("bootstrap failed to execute.");
    } else if (ret != 0) {
      std::stringstream buf;
      buf << "bootstrap died with signal " << ret
          << "; make sure autoconf, automake and libtool are installed";
      if (!verbose) {
        buf << ", see bootstrap.log for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void bi::Driver::configure() {
  auto buildDir = fs::path("build") / suffix();
  if (newBootstrap || newConfigure || newMake ||
      !exists(buildDir / "Makefile")) {
    CWD cwd(buildDir);

    /* compile and link flags */
    std::stringstream cppflags, cflags, cxxflags, ldflags, options, cmd;
    if (arch == "js") {
      //
    } else if (arch == "wasm") {
      cflags << " -s WASM=1";
      cxxflags << " -s WASM=1";
    } else if (arch == "native") {
      //cflags << " -march=native";
      //cxxflags << " -march=native";
      //^ can cause cross-compile issues, e.g. on Open Build Service
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
    } else {
      throw DriverException(
          "unknown architecture '" + arch
              + "'; valid values are 'native', 'js' and 'wasm'");
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
    if (mode == "debug" || mode == "test") {
      options << " --enable-assert";
    } else {
      options << " --disable-assert";
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
    if (mode != "release") {
      options << " SUFFIX=_" << mode;
    }
    options << " MODE=" << mode;

    /* command */
    if (arch == "js" || arch == "wasm") {
      cmd << "emconfigure ";
    }
    cmd << (fs::path("..") / ".." / "configure") << ' ' << options.str();
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > configure.log 2>&1";
    }

    int ret = std::system(cmd.str().c_str());
    if (ret == -1) {
      if (verbose) {
        std::cerr << explain(cmd.str()) << std::endl;
      }
      throw DriverException("configure failed to execute.");
    } else if (ret != 0) {
      std::stringstream buf;
      buf << "configure died with signal " << ret
          << "; make sure all dependencies are installed";
      if (!verbose) {
        buf << ", see " << (buildDir / "configure.log").string() << " and "
            << (buildDir / "config.log").string() << " for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void bi::Driver::target(const std::string& cmd) {
  auto buildDir = fs::canonical(fs::path("build") / "latest");
  CWD cwd(buildDir);

  /* command */
  std::stringstream buf;
  if (arch == "js" || arch == "wasm") {
    buf << "emmake";
  }
  buf << "make";

  /* concurrency */
  if (jobs > 1) {
    buf << " -j " << jobs;
  }

  /* target */
  buf << ' ' << cmd;

  /* strip warnings/notes? */
  buf << " 2>&1";
  if (!warnings) {
    buf << " | grep --line-buffered -v 'warning:'";
  }
  if (!notes) {
    buf << " | grep --line-buffered -v 'note:'";
  }

  /* strip namespaces, which are meant to be internal */
  buf << " | sed -lE 's/[a-zA-Z0-9_]+:://g'";

  /* replace some operators */
  buf << " | sed -lE 's/operator->/./g'";

  /* strip suggestions that reveal internal workings */
  buf << " | sed -lE \"s/; did you mean '[[:alnum:]_]+_'\\?/./\"";
  buf << " | sed -lE \"/note: '[[:alnum:]_]+_' declared here/d\"";
  buf << " | sed -lE \"s/'='/'<-'/\"";
  buf << " | grep --line-buffered -v 'note: expanded from macro'";
  buf << " 1>&2";

  /* handle output */
  std::string log = cmd + ".log";
  if (verbose) {
    std::cerr << buf.str() << std::endl;
  } else {
    buf << " > " << log << " 2>&1";
  }

  int ret = std::system(buf.str().c_str());
  if (ret == -1) {
    if (verbose) {
      std::cerr << explain(buf.str()) << std::endl;
    }
    buf.str("make ");
    buf << cmd;
    if (ret == -1) {
      buf << " failed to execute";
    } else {
      buf << " died with signal " << ret;
    }
    if (!verbose) {
      buf << ", see " << (buildDir / log).string() << " for details.";
    }
    buf << '.';
    throw DriverException(buf.str());
  }
}

void bi::Driver::ldconfig() {
  #ifndef __APPLE__
  auto euid = geteuid();
  if (euid == 0) {
    [[maybe_unused]] int result = std::system("ldconfig");
  }
  #endif
}

const char* bi::Driver::explain(const std::string& cmd) {
  #ifdef HAVE_LIBEXPLAIN_SYSTEM_H
  return explain_system(cmd.c_str());
  #else
  return "";
  #endif
}

std::string bi::Driver::suffix() const {
  /* the suffix is built by joining all build options, in a prescribed order,
   * joined by spaces, then encoding in base 32 */
  std::stringstream buf;
  buf << mode;
  buf << '_' << unit;
  buf << '_' << arch;
  if (staticLib) {
    buf << "_static";
  }
  if (sharedLib) {
    buf << "_shared";
  }
  if (openmp) {
    buf << "_openmp";
  }
  return buf.str();
}

bi::Package* bi::Driver::createPackage(bool includeRequires) {
  Package* package = new Package(packageName);
  if (includeRequires) {
    for (auto name : metaFiles["require.package"]) {
      /* add *.bih dependency */
      fs::path header = tarname(name.string());
      header.replace_extension(".bih");
      package->addHeader(find(includeDirs, header).string());
    }
  }
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".bi") == 0) {
      package->addSource(file.string());
    }
  }
  return package;
}

void bi::Driver::readFiles(const boost::property_tree::ptree& meta,
    const std::string& key, bool checkExists) {
  auto files = meta.get_child_optional(key);
  if (files) {
    for (auto file : files.get()) {
      if (auto str = file.second.get_value_optional<std::string>()) {
        if (str) {
          auto filePath = fs::path(str.get());
          auto fileStr = filePath.string();
          if (checkExists && !exists(filePath)) {
            warn(fileStr + " in META.json does not exist.");
          }
          if (std::regex_search(fileStr,
              std::regex("\\s", std::regex_constants::ECMAScript))) {
            throw DriverException(
                std::string("file name ") + fileStr
                    + " in META.json contains whitespace, which is not supported.");
          }
          auto inserted = allFiles.insert(filePath);
          if (!inserted.second) {
            warn(fileStr + " repeated in META.json.");
          }
          metaFiles[key].push_back(filePath);
        }
      }
    }
  }
}
