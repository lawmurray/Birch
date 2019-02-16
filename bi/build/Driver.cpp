/**
 * @file
 */
#include "Driver.hpp"

#include "bi/build/Compiler.hpp"
#include "bi/build/misc.hpp"
#include "bi/io/md_ostream.hpp"
#include "bi/primitive/encode.hpp"
#include "bi/exception/DriverException.hpp"

#include "boost/algorithm/string.hpp"

#include <getopt.h>
#include <dlfcn.h>

bi::Driver::Driver(int argc, char** argv) :
    /* keep these paths relative, or at least call configure with a
     * relative path from the build directory to the work directory,
     * otherwise a work directory containing spaces causes problems */
    work_dir("."),
    arch("native"),
    prefix(""),
    packageName("Untitled"),
    unity(false),
    staticLib(false),
    sharedLib(true),
    openmp(true),
    warnings(true),
    debug(true),
    verbose(true),
    lazyDeepClone(true),
    cloneMemo(true),
    ancestryMemo(true),
    cloneMemoInitialSize(64),
    cloneMemoDelta(2),
    ancestryMemoInitialSize(8),
    ancestryMemoDelta(2),
    newAutogen(false),
    newConfigure(false),
    newMake(false) {
  enum {
    WORK_DIR_ARG = 256,
    SHARE_DIR_ARG,
    INCLUDE_DIR_ARG,
    LIB_DIR_ARG,
    ARCH_ARG,
    PREFIX_ARG,
    NAME_ARG,
    ENABLE_UNITY_ARG,
    DISABLE_UNITY_ARG,
    ENABLE_STATIC_ARG,
    DISABLE_STATIC_ARG,
    ENABLE_SHARED_ARG,
    DISABLE_SHARED_ARG,
    ENABLE_OPENMP_ARG,
    DISABLE_OPENMP_ARG,
    ENABLE_WARNINGS_ARG,
    DISABLE_WARNINGS_ARG,
    ENABLE_DEBUG_ARG,
    DISABLE_DEBUG_ARG,
    ENABLE_VERBOSE_ARG,
    DISABLE_VERBOSE_ARG,
    ENABLE_LAZY_DEEP_CLONE_ARG,
    DISABLE_LAZY_DEEP_CLONE_ARG,
    ENABLE_CLONE_MEMO_ARG,
    DISABLE_CLONE_MEMO_ARG,
    ENABLE_ANCESTRY_MEMO_ARG,
    DISABLE_ANCESTRY_MEMO_ARG,
    CLONE_MEMO_INITIAL_SIZE_ARG,
    CLONE_MEMO_DELTA_ARG,
    ANCESTRY_MEMO_INITIAL_SIZE_ARG,
    ANCESTRY_MEMO_DELTA_ARG
  };

  int c, option_index;
  option long_options[] = {
      { "work-dir", required_argument, 0, WORK_DIR_ARG },
      { "share-dir", required_argument, 0, SHARE_DIR_ARG },
      { "include-dir", required_argument, 0, INCLUDE_DIR_ARG },
      { "lib-dir", required_argument, 0, LIB_DIR_ARG },
      { "arch", required_argument, 0, ARCH_ARG },
      { "prefix", required_argument, 0, PREFIX_ARG },
      { "name", required_argument, 0, NAME_ARG },
      { "enable-unity", no_argument, 0, ENABLE_UNITY_ARG },
      { "disable-unity", no_argument, 0, DISABLE_UNITY_ARG },
      { "enable-static", no_argument, 0, ENABLE_STATIC_ARG },
      { "disable-static", no_argument, 0, DISABLE_STATIC_ARG },
      { "enable-shared", no_argument, 0, ENABLE_SHARED_ARG },
      { "disable-shared", no_argument, 0, DISABLE_SHARED_ARG },
      { "enable-openmp", no_argument, 0, ENABLE_OPENMP_ARG },
      { "disable-openmp", no_argument, 0, DISABLE_OPENMP_ARG },
      { "enable-warnings", no_argument, 0, ENABLE_WARNINGS_ARG },
      { "disable-warnings", no_argument, 0, DISABLE_WARNINGS_ARG },
      { "enable-debug", no_argument, 0, ENABLE_DEBUG_ARG },
      { "disable-debug", no_argument, 0, DISABLE_DEBUG_ARG },
      { "enable-verbose", no_argument, 0, ENABLE_VERBOSE_ARG },
      { "disable-verbose", no_argument, 0, DISABLE_VERBOSE_ARG },
      { "enable-lazy-deep-clone", no_argument, 0, ENABLE_LAZY_DEEP_CLONE_ARG },
      { "disable-lazy-deep-clone", no_argument, 0, DISABLE_LAZY_DEEP_CLONE_ARG },
      { "enable-clone-memo", no_argument, 0, ENABLE_CLONE_MEMO_ARG },
      { "disable-clone-memo", no_argument, 0, DISABLE_CLONE_MEMO_ARG },
      { "enable-ancestry-memo", no_argument, 0, ENABLE_ANCESTRY_MEMO_ARG },
      { "disable-ancestry-memo", no_argument, 0, DISABLE_ANCESTRY_MEMO_ARG },
      { "clone-memo-initial-size", required_argument, 0, CLONE_MEMO_INITIAL_SIZE_ARG },
      { "clone-memo-delta", required_argument, 0, CLONE_MEMO_DELTA_ARG },
      { "ancestry-memo-initial-size", required_argument, 0, ANCESTRY_MEMO_INITIAL_SIZE_ARG },
      { "ancestry-memo-delta", required_argument, 0, ANCESTRY_MEMO_DELTA_ARG },
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
    case WORK_DIR_ARG:
      work_dir = optarg;
      break;
    case SHARE_DIR_ARG:
      share_dirs.push_back(optarg);
      break;
    case INCLUDE_DIR_ARG:
      include_dirs.push_back(optarg);
      break;
    case LIB_DIR_ARG:
      lib_dirs.push_back(optarg);
      break;
    case ARCH_ARG:
      arch = optarg;
      break;
    case PREFIX_ARG:
      prefix = optarg;
      break;
    case NAME_ARG:
      packageName = optarg;
      break;
    case ENABLE_UNITY_ARG:
      unity = true;
      break;
    case DISABLE_UNITY_ARG:
      unity = false;
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
    case ENABLE_DEBUG_ARG:
      debug = true;
      break;
    case DISABLE_DEBUG_ARG:
      debug = false;
      break;
    case ENABLE_VERBOSE_ARG:
      verbose = true;
      break;
    case DISABLE_VERBOSE_ARG:
      verbose = false;
      break;
    case ENABLE_LAZY_DEEP_CLONE_ARG:
      lazyDeepClone = true;
      break;
    case DISABLE_LAZY_DEEP_CLONE_ARG:
      lazyDeepClone = false;
      break;
    case ENABLE_CLONE_MEMO_ARG:
      cloneMemo = true;
      break;
    case DISABLE_CLONE_MEMO_ARG:
      cloneMemo = false;
      break;
    case ENABLE_ANCESTRY_MEMO_ARG:
      ancestryMemo = true;
      break;
    case DISABLE_ANCESTRY_MEMO_ARG:
      ancestryMemo = false;
      break;
    case CLONE_MEMO_INITIAL_SIZE_ARG:
      cloneMemoInitialSize = atoi(optarg);
      break;
    case CLONE_MEMO_DELTA_ARG:
      cloneMemoDelta = atoi(optarg);
      break;
    case ANCESTRY_MEMO_INITIAL_SIZE_ARG:
      ancestryMemoInitialSize = atoi(optarg);
      break;
    case ANCESTRY_MEMO_DELTA_ARG:
      ancestryMemoDelta = atoi(optarg);
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
  if (!isPower2(cloneMemoInitialSize)) {
    throw DriverException("--clone-memo-initial-size must be a positive power of 2.");
  }
  if (cloneMemoDelta <= 0) {
    throw DriverException("--clone-memo-delta must be a positive integer.");
  }
  if (!isPower2(ancestryMemoInitialSize)) {
    throw DriverException("--ancestry-memo-initial-size must be a positive power of 2.");
  }
  if (ancestryMemoDelta <= 0) {
    throw DriverException("--ancestry-memo-delta must be a positive integer.");
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
      share_dirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    share_dirs.push_back(fs::path(prefix) / "share");
  }
  #ifdef DATADIR
  share_dirs.push_back(fs::path(STRINGIFY(DATADIR)) / "birch");
  #endif

  /* include dirs */
  include_dirs.push_back(work_dir);
  include_dirs.push_back(work_dir / "build" / suffix());
  if (BIRCH_INCLUDE_PATH) {
    std::stringstream birch_include_path(BIRCH_INCLUDE_PATH);
    while (std::getline(birch_include_path, input, ':')) {
      include_dirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    include_dirs.push_back(fs::path(prefix) / "include");
  }
  #ifdef INCLUDEDIR
  include_dirs.push_back(STRINGIFY(INCLUDEDIR));
  #endif

  /* lib dirs */
  if (BIRCH_LIBRARY_PATH) {
    std::stringstream birch_library_path(BIRCH_LIBRARY_PATH);
    while (std::getline(birch_library_path, input, ':')) {
      lib_dirs.push_back(input);
    }
  }
  if (!prefix.empty()) {
    lib_dirs.push_back(fs::path(prefix) / "lib");
  }
  #ifdef LIBDIR
  lib_dirs.push_back(STRINGIFY(LIBDIR));
  #endif
}

void bi::Driver::run(const std::string& prog) {
  /* get package information */
  meta();

  /* dynamically load possible programs */
  typedef void prog_t(int argc, char** argv);

  void* handle;
  void* addr;
  char* msg;
  prog_t* fcn;

  fs::path so = std::string("lib") + tarname(packageName);
#ifdef __APPLE__
  so.replace_extension(".dylib");
#else
  so.replace_extension(".so");
#endif

  /* look in built libs first */
  handle = dlopen(so.c_str(), RTLD_NOW);
  msg = dlerror();
  if (handle == NULL) {
    std::stringstream buf;
    buf << "Could not load " << so.string() << ", " << msg << '.';
    throw DriverException(buf.str());
  } else {
    addr = dlsym(handle, prog.c_str());
    msg = dlerror();
    if (msg != NULL) {
      std::stringstream buf;
      buf << "Could not find symbol " << prog << " in " << so.string() << '.';
      throw DriverException(buf.str());
    } else {
      fcn = reinterpret_cast<prog_t*>(addr);
      fcn(largv.size(), largv.data());
    }
    dlclose(handle);
  }
}

void bi::Driver::build() {
  meta();
  setup();
  compile();
  autogen();
  configure();
  target();
  if (arch == "js" || arch == "wasm") {
    target("birch.html");
  }
}

void bi::Driver::install() {
  meta();
  setup();
  compile();
  autogen();
  configure();
  target("install");
}

void bi::Driver::uninstall() {
  meta();
  setup();
  compile();
  autogen();
  configure();
  target("uninstall");
}

void bi::Driver::dist() {
  meta();
  setup();
  compile();
  autogen();
  configure();
  target("dist");
}

void bi::Driver::clean() {
  meta();
  setup();

  auto previous_dir = fs::current_path();
  fs::current_path(work_dir);

  fs::remove_all("build");
  fs::remove_all("autom4te.cache");
  fs::remove_all("m4");
  fs::remove("aclocal.m4");
  fs::remove("autogen.log");
  fs::remove("autogen.sh");
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

  fs::current_path(previous_dir);
}

void bi::Driver::tune() {
  meta();

  verbose = false;  // makes things tidier
  unity = true;     // makes compile faster
  debug = false;    // makes run faster

  /* best times */
  double bestEager, bestLazy;

  /* best options */
  int bestEagerCloneMemoInitialSize;
  bool bestLazyCloneMemo;
  int bestLazyCloneMemoInitialSize;
  int bestLazyCloneMemoDelta;
  bool bestLazyAncestryMemo;
  int bestLazyAncestryMemoInitialSize;
  int bestLazyAncestryMemoDelta;

  /* proposed initial sizes, to test */
  auto initialSizes = { 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

  /* proposed deltas, to test */
  auto deltas = { 1, 2, 4, 8, 16, 32 };

  /* best eager configuration */
  std::cerr << "setting --disable-lazy-deep-clone" << std::endl;
  lazyDeepClone = false;

  std::cerr << "trying --clone-memo-initial-size" << std::endl;
  std::tie(bestEagerCloneMemoInitialSize, bestEager) = choose(&cloneMemoInitialSize, initialSizes);

  /* best lazy configuration */
  std::cerr << "setting --enable-lazy-deep-clone" << std::endl;
  lazyDeepClone = true;

  std::cerr << "trying --clone-memo-initial-size" << std::endl;
  std::tie(bestLazyCloneMemoInitialSize, bestLazy) = choose(&cloneMemoInitialSize, initialSizes);

  std::cerr << "trying --enable-clone-memo" << std::endl;
  std::tie(bestLazyCloneMemo, bestLazy) = choose(&cloneMemo, { true, false });
  if (bestLazyCloneMemo) {
    std::cerr << "trying --clone-memo-initial-size" << std::endl;
    std::tie(bestLazyCloneMemoInitialSize, bestLazy) = choose(&cloneMemoInitialSize, initialSizes);
    std::cerr << "trying --clone-memo-delta" << std::endl;
    std::tie(bestLazyCloneMemoDelta, bestLazy) = choose(&cloneMemoDelta, deltas);
  }

  std::cerr << "trying --enable-ancestry-memo" << std::endl;
  std::tie(bestLazyAncestryMemo, bestLazy) = choose(&ancestryMemo, { true, false });
  if (bestLazyAncestryMemo) {
    std::cerr << "trying --ancestry-memo-initial-size" << std::endl;
    std::tie(bestLazyAncestryMemoInitialSize, bestLazy) = choose(&ancestryMemoInitialSize, initialSizes);
    std::cerr << "trying --ancestry-memo-delta" << std::endl;
    std::tie(bestLazyAncestryMemoDelta, bestLazy) = choose(&ancestryMemoDelta, deltas);
  }

  /* choose one or the other and report */
  if (bestEager < bestLazy) {
    std::cout << "suggested:";
    std::cout << " --disable-lazy-deep-clone";
    std::cout << " --clone-memo-initial-size=" << bestEagerCloneMemoInitialSize;
  } else {
    std::cout << "suggested:";
    std::cout << " --enable-lazy-deep-clone";
    if (bestLazyCloneMemo) {
      std::cout << " --enable-clone-memo";
      std::cout << " --clone-memo-initial-size=" << bestLazyCloneMemoInitialSize;
      std::cout << " --clone-memo-delta=" << bestLazyCloneMemoDelta;
    } else {
      std::cout << " --clone-memo-initial-size=" << bestLazyCloneMemoInitialSize;
      std::cout << " --disable-clone-memo";
    }
    if (bestLazyAncestryMemo) {
      std::cout << " --enable-ancestry-memo";
      std::cout << " --ancestry-memo-initial-size=" << bestLazyAncestryMemoInitialSize;
      std::cout << " --ancestry-memo-delta=" << bestLazyAncestryMemoDelta;
    } else {
      std::cout << " --disable-ancestry-memo";
    }
  }
  std::cout << std::endl;
}

double bi::Driver::time() {
  for (auto name : metaFiles["require.package"]) {
    Driver driver(*this);
    driver.work_dir = work_dir / ".." / name;
    driver.install();
  }
  Driver driver(*this);
  driver.install();

  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
  driver.run("run_");
  std::chrono::time_point<std::chrono::system_clock> stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  return elapsed.count();
}

void bi::Driver::init() {
  auto previous_dir = fs::current_path();
  fs::current_path(work_dir);

  fs::create_directory("bi");
  fs::create_directory("input");
  fs::create_directory("output");
  copy_with_prompt(find(share_dirs, "gitignore"), ".gitignore");
  copy_with_prompt(find(share_dirs, "LICENSE"), "LICENSE");

  std::string contents;

  contents = read_all(find(share_dirs, "META.json"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream metaStream("META.json");
  if (metaStream.fail()) {
    std::stringstream buf;
    buf << "Could not open META.json for writing.";
    throw DriverException(buf.str());
  }
  metaStream << contents;

  contents = read_all(find(share_dirs, "README.md"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream readmeStream("README.md");
  if (readmeStream.fail()) {
    std::stringstream buf;
    buf << "Could not open README.md for writing.";
    throw DriverException(buf.str());
  }
  readmeStream << contents;

  fs::current_path(work_dir);
}

void bi::Driver::check() {
  auto previous_dir = fs::current_path();
  fs::current_path(work_dir);

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
  interesting.insert(".m");
  interesting.insert(".R");
  interesting.insert(".json");
  interesting.insert(".ubj");
  interesting.insert(".cpp");
  interesting.insert(".hpp");

  exclude.insert("autogen.sh");
  exclude.insert("ltmain.sh");

  fs::recursive_directory_iterator iter("."), end;
  while (iter != end) {
    auto path = remove_first(iter->path());
    auto name = path.filename().string();
    auto ext = path.extension().string();
    if (path.string() == "build" || path.string() == "output" ||
        path.string() == "site") {
      iter.no_push();
    } else if (interesting.find(ext) != interesting.end()
        && exclude.find(name) == exclude.end()) {
      if (allFiles.find(path.string()) == allFiles.end()) {
        warn(std::string("is ") + path.string()
                + " missing from META.json file?");
      }
    }
    ++iter;
  }

  fs::current_path(previous_dir);
}

void bi::Driver::docs() {
  meta();

  auto previous_dir = fs::current_path();
  fs::current_path(work_dir);

  Package* package = createPackage();

  /* parse all files */
  Compiler compiler(package, fs::path("build") / suffix(), unity);
  compiler.parse();
  compiler.resolve();

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
  mkdocsStream << "  name: 'readthedocs'\n";
  mkdocsStream << "markdown_extensions:\n";
  mkdocsStream << "  - admonition\n";
  mkdocsStream << "  - mdx_math:\n";
  mkdocsStream << "      enable_dollar_delimiter: True\n";
  mkdocsStream << "extra_javascript:\n";
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
    docsStream << packageDesc << '\n';
    docsStream.close();
  }
  mkdocsStream << "  - index.md\n";

  std::string str = read_all("DOCS.md");
  std::regex reg("(?:^|\r?\n)(##?) (.*?)(?=\r?\n|$)", std::regex_constants::ECMAScript);
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

  fs::current_path(previous_dir);
  delete package;
}

void bi::Driver::meta() {
  /* clear any previous read */
  packageName = "Untitled";
  packageDesc = "";
  metaFiles.clear();
  allFiles.clear();

  auto previous_dir = fs::current_path();
  fs::current_path(work_dir);

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
  if (auto desc = meta.get_optional<std::string>("description")) {
    packageDesc = desc.get();
  }

  /* external requirements */
  if (packageName != "Birch.Standard") {
    /* implicitly include the standard library, if this package is not,
     * itself, the standard library */
    metaFiles["require.package"].push_front("Birch.Standard");
  }
  readFiles(meta, "require.package", false);
  readFiles(meta, "require.header", false);
  readFiles(meta, "require.library", false);
  readFiles(meta, "require.program", false);

  /* convert package requirements to header and library requirements */
  for (auto name : metaFiles["require.package"]) {
    auto internalName = tarname(name.string());
    auto header = fs::path("bi") / internalName;
    header.replace_extension(".hpp");
    metaFiles["require.header"].push_back(header.string());
    metaFiles["require.library"].push_back(internalName);
  }

  /* manifest */
  readFiles(meta, "manifest.header", true);
  readFiles(meta, "manifest.source", true);
  readFiles(meta, "manifest.data", true);
  readFiles(meta, "manifest.other", true);

  fs::current_path(previous_dir);
}

void bi::Driver::setup() {
  auto previous_dir = fs::current_path();
  auto build_dir = fs::path("build") / suffix();
  fs::current_path(work_dir);

  /* internal name of package */
  auto internalName = tarname(packageName);

  /* create build directory */
  if (!fs::exists(build_dir)) {
    if (!fs::create_directories(build_dir)) {
      std::stringstream buf;
      buf << "could not create build directory " << build_dir << '.';
      throw DriverException(buf.str());
    }

    /* workaround for error given by some versions of autotools, "Something
     * went wrong bootstrapping makefile fragments for automatic dependency
     * tracking..." */
    fs::create_directories(build_dir / "bi" / ".deps");
    fs::ofstream(build_dir / "bi" / ".deps" / (internalName + ".gch.Plo"));
  }

  /* copy build files into build directory */
  newAutogen = copy_if_newer(find(share_dirs, "autogen.sh"), "autogen.sh");
  fs::permissions("autogen.sh", fs::add_perms | fs::owner_exe);

  fs::path m4_dir("m4");
  if (!fs::exists(m4_dir)) {
    if (!fs::create_directory(m4_dir)) {
      std::stringstream buf;
      buf << "Could not create m4 directory " << m4_dir << '.';
      throw DriverException(buf.str());
    }
  }
  copy_if_newer(find(share_dirs, "ax_cxx_compile_stdcxx.m4"),
      m4_dir / "ax_cxx_compile_stdcxx.m4");
  copy_if_newer(find(share_dirs, "ax_check_define.m4"),
      m4_dir / "ax_check_define.m4");
  copy_if_newer(find(share_dirs, "ax_gcc_builtin.m4"),
      m4_dir / "ax_gcc_builtin.m4");

  /* update configure.ac */
  std::string contents = read_all(find(share_dirs, "configure.ac"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_TARNAME", internalName);
  std::stringstream configureStream;
  configureStream << contents << "\n\n";

  /* required headers */
  if (!metaFiles["require.header"].empty()) {
    configureStream << "if test x$emscripten = xfalse; then\n";
  }
  for (auto file : metaFiles["require.header"]) {
    configureStream << "  AC_CHECK_HEADERS([" << file.string() << "], [], "
        << "[AC_MSG_ERROR([header required by " << packageName
        << " package not found.])], [-])\n";
  }
  if (!metaFiles["require.header"].empty()) {
    configureStream << "fi\n";
  }

  /* required libraries */
  for (auto file : metaFiles["require.library"]) {
    configureStream << "  AC_CHECK_LIB([" << file.string() << "], [main], "
        << "[], [AC_MSG_ERROR([library required by " << packageName
        << " package not found.])])\n";
  }

  /* required programs */
  for (auto file : metaFiles["require.program"]) {
    configureStream << "  AC_PATH_PROG([PROG], [" << file.string()
                  << "], [])\n";
    configureStream << "  if test \"$PROG\" = \"\"; then\n";
    configureStream << "    AC_MSG_ERROR([" << file.string() << " program "
        << "required by " << packageName << " package not found.])\n";
    configureStream << "  fi\n";
  }

  /* footer */
  configureStream << "AC_CONFIG_FILES([Makefile])\n";
  configureStream << "AC_OUTPUT\n";

  newConfigure = write_all_if_different("configure.ac", configureStream.str());

  /* update Makefile.am */
  contents = read_all(find(share_dirs, "Makefile.am"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  boost::replace_all(contents, "PACKAGE_TARNAME", internalName);

  std::stringstream makeStream;
  makeStream << contents << "\n\n";
  makeStream << "lib_LTLIBRARIES = lib" << internalName << ".la\n\n";

  /* sources derived from *.bi files */
  makeStream << "nodist_lib" << internalName << "_la_SOURCES =";
  makeStream << " \\\n  bi/" << internalName << ".cpp";
  if (!unity) {
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".bi") == 0) {
        fs::path cppFile = file;
        cppFile.replace_extension(".cpp");
        makeStream << " \\\n  " << cppFile.string();
      }
    }
  }
  makeStream << '\n';

  /* other *.cpp files */
  makeStream << "lib" << internalName << "_la_SOURCES = ";
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".cpp") == 0
        || file.extension().compare(".c") == 0) {
      makeStream << " \\\n  " << file.string();
    }
  }
  makeStream << '\n';

  /* headers to install and distribute */
  makeStream << "nobase_include_HEADERS =";
  makeStream << " \\\n  bi/" << internalName << ".hpp";
  makeStream << " \\\n  bi/" << internalName << ".bih";
  for (auto file : metaFiles["manifest.header"]) {
    if (file.extension().compare(".hpp") == 0
        || file.extension().compare(".h") == 0) {
      makeStream << " \\\n  " << file.string();
    }
  }
  makeStream << '\n';

  /* data files to distribute */
  makeStream << "dist_pkgdata_DATA = ";
  for (auto file : metaFiles["manifest.data"]) {
    makeStream << " \\\n  " << file.string();
  }
  makeStream << '\n';

  /* other files to not distribute */
  makeStream << "noinst_DATA = ";
  for (auto file : metaFiles["manifest.other"]) {
    makeStream << " \\\n  " << file.string();
  }
  makeStream << '\n';

  newMake = write_all_if_different("Makefile.am", makeStream.str());

  fs::current_path(previous_dir);
}

bi::Package* bi::Driver::createPackage() {
  Package* package = new Package(packageName);
  for (auto name : metaFiles["require.package"]) {
    /* add *.bih dependency */
    fs::path header = fs::path("bi") / tarname(name.string());
    header.replace_extension(".bih");
    package->addHeader(find(include_dirs, header).string());
  }
  for (auto file : metaFiles["manifest.source"]) {
    if (file.extension().compare(".bi") == 0) {
      package->addSource(file.string());
    }
  }
  return package;
}

void bi::Driver::compile() {
  Package* package = createPackage();

  auto previous_dir = fs::current_path();
  auto build_dir = fs::path("build") / suffix();
  fs::current_path(work_dir);

  Compiler compiler(package, build_dir, unity);
  compiler.parse();
  compiler.resolve();
  compiler.gen();

  fs::current_path(previous_dir);
  delete package;
}

void bi::Driver::autogen() {
  if (newAutogen || newConfigure || newMake
      || !fs::exists(work_dir / "configure")
      || !fs::exists(work_dir / "install-sh")) {
    auto previous_dir = fs::current_path();
    fs::current_path(work_dir);

    std::stringstream cmd;
    cmd << (fs::path(".") / "autogen.sh");
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > autogen.log 2>&1";
    }

    int ret = system(cmd.str().c_str());
    if (ret == -1) {
      throw DriverException("autogen.sh failed to execute.");
    } else if (ret != 0) {
      std::stringstream buf;
      buf << "autogen.sh died with signal " << ret
          << "; make sure autoconf, automake and libtool are installed";
      if (!verbose) {
        buf << ", see " << (work_dir / "autogen.log").string()
            << " for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
    fs::current_path(previous_dir);
  }
}

void bi::Driver::configure() {
  auto build_dir = work_dir / "build" / suffix();
  if (newAutogen || newConfigure || newMake
      || !exists(build_dir / "Makefile")) {
    auto previous_dir = fs::absolute(fs::current_path());
    fs::current_path(build_dir);

    /* compile and link flags */
    std::stringstream cppflags, cflags, cxxflags, ldflags, options, cmd;
    if (arch == "js") {
      //
    } else if (arch == "wasm") {
      cflags << " -s WASM=1";
      cxxflags << " -s WASM=1";
    } else if (arch == "native") {
      cflags << " -march=native";
      cxxflags << " -march=native";
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
      throw DriverException("unknown architecture '" + arch +
          "'; valid values are 'native', 'js' and 'wasm'");
    }
    if (warnings) {
      cflags << " -Wall";
      cxxflags << " -Wall";
    }
    if (debug) {
      //@todo Consider a development build with these settings
      //cflags << " -O0 -fno-inline -g";
      //cxxflags << " -O0 -fno-inline -g";
      cflags << " -Og -g";
      cxxflags << " -Og -g";
    } else {
      cppflags << " -DNDEBUG";
      cflags << " -O3 -funroll-loops -flto -g";
      cxxflags << " -O3 -funroll-loops -flto -g";
    }

    /* defines */
    if (lazyDeepClone) {
      cppflags << " -DENABLE_LAZY_DEEP_CLONE=1";
    } else {
      cppflags << " -DENABLE_LAZY_DEEP_CLONE=0";
    }
    if (cloneMemo) {
      cppflags << " -DENABLE_CLONE_MEMO=1";
    } else {
      cppflags << " -DENABLE_CLONE_MEMO=0";
    }
    if (ancestryMemo) {
      cppflags << " -DENABLE_ANCESTRY_MEMO=1";
    } else {
      cppflags << " -DENABLE_ANCESTRY_MEMO=0";
    }
    cppflags << " -DCLONE_MEMO_INITIAL_SIZE=" << cloneMemoInitialSize;
    cppflags << " -DCLONE_MEMO_DELTA=" << cloneMemoDelta;
    cppflags << " -DANCESTRY_MEMO_INITIAL_SIZE=" << ancestryMemoInitialSize;
    cppflags << " -DANCESTRY_MEMO_DELTA=" << ancestryMemoDelta;

    /* include path */
    for (auto iter = include_dirs.begin(); iter != include_dirs.end();
        ++iter) {
      cppflags << " -I" << iter->string();
    }
    for (auto iter = lib_dirs.begin(); iter != lib_dirs.end();
        ++iter) {
      ldflags << " -L" << iter->string();
    }

    /* library path */
    for (auto iter = lib_dirs.begin(); iter != lib_dirs.end();
        ++iter) {
      ldflags << " -Wl,-rpath," << iter->string();
    }

    /* configure options */
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

    /* command */
    if (arch == "js" || arch == "wasm") {
      cmd << "emconfigure ";
    }
    cmd << (fs::path("..") / ".." / "configure") << " " << options.str();
    // ^ build dir is work_dir/build/suffix, so configure script two dirs up
    if (!cppflags.str().empty()) {
      cmd << " CPPFLAGS=\"" << cppflags.str() << "\"";
    }
    if (!cflags.str().empty()) {
      cmd << " CFLAGS=\"" << cflags.str() << "\"";
    }
    if (!cxxflags.str().empty()) {
      cmd << " CXXFLAGS=\"" << cxxflags.str() << "\"";
    }
    if (!ldflags.str().empty()) {
      cmd << " LDFLAGS=\"" << ldflags.str() << "\"";
    }
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > configure.log 2>&1";
    }

    int ret = system(cmd.str().c_str());
    if (ret == -1) {
      throw DriverException("configure failed to execute.");
    } else if (ret != 0) {
      std::stringstream buf;
      buf << "configure died with signal " << ret
          << "; make sure all dependencies are installed";
      if (!verbose) {
        buf << ", see " << (build_dir / "configure.log").string() << " and "
            << (build_dir / "config.log").string() << " for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }

    /* change back to original working dir */
    fs::current_path(previous_dir);
  }
}

void bi::Driver::target(const std::string& cmd) {
  auto previous_dir = fs::absolute(fs::current_path());
  auto build_dir = work_dir / "build" / suffix();
  fs::current_path(build_dir);

  /* command */
  std::stringstream buf;
  if (arch == "js" || arch == "wasm") {
    buf << "emmake";
  }
  buf << "make";

  /* concurrency */
  unsigned ncores = std::thread::hardware_concurrency();
  if (ncores > 1) {
    buf << " -j " << ncores;
  }

  /* target */
  buf << ' ' << cmd;

  /* handle output */
  std::string log = cmd + ".log";
  if (verbose) {
    std::cerr << buf.str() << std::endl;
  } else {
    buf << " > " << log << " 2>&1";
  }

  int ret = system(buf.str().c_str());
  if (ret != 0) {
    buf.str("make ");
    buf << cmd;
    if (ret == -1) {
      buf << " failed to execute";
    } else {
      buf << " died with signal " << ret;
    }
    if (!verbose) {
      buf << ", see " << (build_dir / log).string() << " for details.";
    }
    buf << '.';
    throw DriverException(buf.str());
  }

  /* change back to original working dir */
  fs::current_path(previous_dir);
}

std::string bi::Driver::suffix() const {
  /* the suffix is built by joining all build options, in a prescribed order,
   * joined by spaces, then encoding in base 32 */
  std::stringstream buf;
  buf << arch << ' ';
  buf << unity << ' ';
  buf << staticLib << ' ';
  buf << sharedLib << ' ';
  buf << openmp << ' ';
  buf << debug << ' ';
  buf << lazyDeepClone << ' ';
  buf << cloneMemo << ' ';
  buf << ancestryMemo << ' ';
  buf << cloneMemoInitialSize << ' ';
  buf << cloneMemoDelta << ' ';
  buf << ancestryMemoInitialSize << ' ';
  buf << ancestryMemoDelta << ' ';
  return encode32(buf.str());
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
          if (checkExists && !exists(work_dir / filePath)) {
            warn(fileStr + " in META.json does not exist.");
          }
          if (std::regex_search(fileStr, std::regex("\\s",
              std::regex_constants::ECMAScript))) {
            throw DriverException(std::string("file name ") + fileStr +
                " in META.json contains whitespace, which is not supported.");
          }
          if (filePath.parent_path().string() == "bi" &&
              filePath.stem().string() == tarname(packageName)) {
            throw DriverException(std::string("file name ") + fileStr +
                " in META.json is the same as the package name, which is not supported.");
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
