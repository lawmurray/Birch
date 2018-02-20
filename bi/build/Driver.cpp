/**
 * @file
 */
#include "Driver.hpp"

#include "bi/build/Compiler.hpp"
#include "bi/build/misc.hpp"
#include "bi/io/md_ostream.hpp"
#include "bi/exception/DriverException.hpp"

#include "boost/algorithm/string.hpp"

#include <getopt.h>
#include <dlfcn.h>

#include <iostream>
#include <regex>
#include <unordered_set>

bi::Driver::Driver(int argc, char** argv) :
    work_dir(fs::current_path()),
    build_dir(fs::current_path() / "build"),
    lib_dir(fs::current_path() / "build" / ".libs"),
    arch("native"),
    prefix(""),
    packageName("Untitled"),
    warnings(true),
    debug(true),
    verbose(true),
    newAutogen(false),
    newConfigure(false),
    newMake(false),
    newMeta(false) {
  enum {
    SHARE_DIR_ARG = 256,
    INCLUDE_DIR_ARG,
    LIB_DIR_ARG,
    ARCH_ARG,
    PREFIX_ARG,
    NAME_ARG,
    ENABLE_WARNINGS_ARG,
    DISABLE_WARNINGS_ARG,
    ENABLE_DEBUG_ARG,
    DISABLE_DEBUG_ARG,
    ENABLE_VERBOSE_ARG,
    DISABLE_VERBOSE_ARG
  };

  int c, option_index;
  option long_options[] = {
      { "share-dir", required_argument, 0, SHARE_DIR_ARG },
      { "include-dir", required_argument, 0, INCLUDE_DIR_ARG },
      { "lib-dir", required_argument, 0, LIB_DIR_ARG },
      { "arch", required_argument, 0, ARCH_ARG },
      { "prefix", required_argument, 0, PREFIX_ARG },
      { "name", required_argument, 0, NAME_ARG },
      { "enable-warnings", no_argument, 0, ENABLE_WARNINGS_ARG },
      { "disable-warnings", no_argument, 0, DISABLE_WARNINGS_ARG },
      { "enable-debug", no_argument, 0, ENABLE_DEBUG_ARG },
      { "disable-debug", no_argument, 0, DISABLE_DEBUG_ARG },
      { "enable-verbose", no_argument, 0, ENABLE_VERBOSE_ARG },
      { "disable-verbose", no_argument, 0, DISABLE_VERBOSE_ARG },
      { 0, 0, 0, 0 }
  };
  const char* short_options = "-";  // treats non-options as short option 1

  /* mutable copy of argv and argc */
  largv.insert(largv.begin(), argv, argv + argc);
  std::vector<char*> fargv;

  /* read options */
  std::vector<char*> unknown;
  opterr = 0;  // handle error reporting ourselves
  c = getopt_long_only(largv.size(), largv.data(), short_options,
      long_options, &option_index);
  while (c != -1) {
    switch (c) {
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

  /* environment variables */
  char* BIRCH_SHARE_PATH = getenv("BIRCH_SHARE_PATH");
  char* BIRCH_INCLUDE_PATH = getenv("BIRCH_INCLUDE_PATH");
  char* BIRCH_LIBRARY_PATH = getenv("BIRCH_LIBRARY_PATH");
  std::string input;

  /* share dirs */
  if (BIRCH_SHARE_PATH) {
    std::stringstream birch_share_path(BIRCH_SHARE_PATH);
    while (std::getline(birch_share_path, input, ':')) {
      share_dirs.push_back(input);
    }
  }
#ifdef DATADIR
  share_dirs.push_back(fs::path(STRINGIFY(DATADIR)) / "birch");
#endif

  /* include dirs */
  include_dirs.push_back(work_dir);
  include_dirs.push_back(build_dir);
  if (BIRCH_INCLUDE_PATH) {
    std::stringstream birch_include_path(BIRCH_INCLUDE_PATH);
    while (std::getline(birch_include_path, input, ':')) {
      include_dirs.push_back(input);
    }
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
  if (exists(lib_dir / so)) {
    so = lib_dir / so;
  }
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
  fs::remove_all(build_dir);
}

void bi::Driver::init() {
  fs::create_directory("bi");
  fs::create_directory("input");
  fs::create_directory("output");
  copy_with_prompt(find(share_dirs, "gitignore"), ".gitignore");
  copy_with_prompt(find(share_dirs, "LICENSE"), "LICENSE");

  std::string contents;

  contents = read_all(find(share_dirs, "META.json"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream metaStream(work_dir / "META.json");
  metaStream << contents;

  contents = read_all(find(share_dirs, "README.md"));
  boost::replace_all(contents, "PACKAGE_NAME", packageName);
  fs::ofstream readmeStream(work_dir / "README.md");
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
    if (path.string() == "build" || path.string() == "results") {
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
}

void bi::Driver::docs() {
  current_path(work_dir);
  meta();
  Package* package = createPackage();

  /* parse all files */
  Compiler compiler(package, work_dir, build_dir);
  compiler.parse();
  compiler.resolve();

  /* output everything into single file */
  fs::ofstream stream("DOCS.md");
  md_ostream output(stream);
  output << package;
  stream.close();

  /* split that file into multiple files for mkdocs */
  fs::ofstream mkdocs("mkdocs.yml");
  mkdocs << "site_name: '" << packageName << "'\n";
  mkdocs << "theme: readthedocs\n";
  mkdocs << "markdown_extensions:\n";
  mkdocs << "  - admonition\n";
  mkdocs << "  - mdx_math:\n";
  mkdocs << "      enable_dollar_delimiter: True\n";
  mkdocs << "extra_javascript:\n";
  mkdocs << "  - 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'\n";
  mkdocs << "pages:\n";

  fs::path docs("docs"), file;
  fs::create_directories(docs);
  fs::create_directories(docs / "programs");
  fs::create_directories(docs / "functions");
  fs::create_directories(docs / "fibers");
  fs::create_directories(docs / "unary_operators");
  fs::create_directories(docs / "binary_operators");
  fs::create_directories(docs / "classes");

  stream.open(docs / "index.md");
  stream << packageDesc << '\n';
  stream.close();
  mkdocs << "  - index.md\n";

  std::string str = read_all("DOCS.md");
  std::regex reg("(?:^|\r?\n)(##?) (.*?)(?=\r?\n|$)", std::regex_constants::ECMAScript);
  std::smatch match;
  std::string str1 = str, h1, h2;
  while (std::regex_search(str1, match, reg)) {
    if (stream.is_open()) {
      stream << match.prefix();
    }
    if (match.str(1) == "#") {
      /* first level header */
      h1 = match.str(2);
      mkdocs << "  - '" << h1 << "': ";

      /* among first-level headers, only variables and types have their own
       * page, rather than being further split into a page per item */
      if (h1 == "Variables" || h1 == "Types") {
        std::string filename = h1;
        boost::to_lower(filename);
        file = fs::path(filename + ".md");
        mkdocs << file.string();
        if (stream.is_open()) {
          stream.close();
        }
        stream.open(docs / file);
        stream << "# " << h1 << "\n\n";
      }
      mkdocs << '\n';
      boost::to_lower(h1);
      boost::replace_all(h1, " ", "_");
    } else {
      /* second level header */
      h2 = match.str(2);
      mkdocs << "    - '" << h2 << "': ";
      file = fs::path(h1) / (h2 + ".md");
      mkdocs << file.string() << "\n";
      if (stream.is_open()) {
        stream.close();
      }
      stream.open(docs / file);
      stream << "## " << h2 << "\n\n";
    }
    str1 = match.suffix();
  }
  if (stream.is_open()) {
    stream << str1;
    stream.close();
  }
  delete package;
}

void bi::Driver::meta() {
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

  /*  manifest */
  readFiles(meta, "manifest.header", true);
  readFiles(meta, "manifest.source", true);
  readFiles(meta, "manifest.data", true);
  readFiles(meta, "manifest.other", true);

  /* external requirements */
  readFiles(meta, "require.package", false);
  readFiles(meta, "require.header", false);
  readFiles(meta, "require.library", false);
  readFiles(meta, "require.program", false);
}

void bi::Driver::setup() {
  /* internal name of package */
  auto internalName = tarname(packageName);

  /* create build directory */
  if (!fs::exists(build_dir)) {
    if (!fs::create_directory(build_dir)) {
      std::stringstream buf;
      buf << "could not create build directory " << build_dir << '.';
      throw DriverException(buf.str());
    }
  }

  /* copy build files into build directory */
  newAutogen = copy_if_newer(find(share_dirs, "autogen.sh"),
      work_dir / "autogen.sh");
  fs::permissions(work_dir / "autogen.sh", fs::add_perms | fs::owner_exe);
  newConfigure = copy_if_newer(find(share_dirs, "configure.ac"),
      work_dir / "configure.ac");
  newMake = copy_if_newer(find(share_dirs, "Makefile.am"),
      work_dir / "Makefile.am");
  newMeta = fs::last_write_time("META.json")
      > last_write_time(work_dir / "Makefile.am");

  fs::path m4_dir = work_dir / "m4";
  if (!fs::exists(m4_dir)) {
    if (!fs::create_directory(m4_dir)) {
      std::stringstream buf;
      buf << "could not create m4 directory " << m4_dir << '.';
      throw DriverException(buf.str());
    }
  }
  copy_if_newer(find(share_dirs, "ax_cxx_compile_stdcxx.m4"),
      m4_dir / "ax_cxx_compile_stdcxx.m4");
  copy_if_newer(find(share_dirs, "ax_check_define.m4"),
      m4_dir / "ax_check_define.m4");

  /* update configure.ac */
  if (newConfigure || newMeta) {
    std::string contents = read_all(find(share_dirs, "configure.ac"));
    boost::replace_all(contents, "PACKAGE_NAME", packageName);
    boost::replace_all(contents, "PACKAGE_TARNAME", internalName);
    fs::ofstream configureStream(work_dir / "configure.ac");
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

    configureStream.close();
  }

  /* update Makefile.am */
  if (newMake || newMeta) {
    std::string contents = read_all(find(share_dirs, "Makefile.am"));
    boost::replace_all(contents, "PACKAGE_NAME", packageName);
    boost::replace_all(contents, "PACKAGE_TARNAME", internalName);

    fs::ofstream makeStream(work_dir / "Makefile.am");
    makeStream << contents << "\n\n";
    makeStream << "lib_LTLIBRARIES = lib" << internalName << ".la\n\n";

    /* *.cpp files */
    makeStream << "lib" << internalName << "_la_SOURCES = ";
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".cpp") == 0
          || file.extension().compare(".c") == 0) {
        makeStream << " \\\n  " << file.string();
      }
    }
    makeStream << '\n';

    /* sources derived from *.bi files */
    makeStream << "nodist_lib" << internalName << "_la_SOURCES = ";
    for (auto file : metaFiles["manifest.source"]) {
      if (file.extension().compare(".bi") == 0) {
        fs::path cppFile = file;
        cppFile.replace_extension(".cpp");
        makeStream << " \\\n  " << cppFile.string();
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

    makeStream.close();
  }
}

bi::Package* bi::Driver::createPackage() {
  Package* package = new Package(packageName);
  if (packageName != "Birch.Standard") {
    /* disable inclusion of the standard library when the project is, itself,
     * the standard library (!) */
    auto header = fs::path("bi") / "birch_standard.bih";
    package->addHeader(find(include_dirs, header).string());
  }
  for (auto name : metaFiles["require.package"]) {
    /* add *.bih and *.hpp header and library dependencies */
    auto internalName = tarname(name.string());
    fs::path header = fs::path("bi") / internalName;
    header.replace_extension(".bih");
    package->addHeader(find(include_dirs, header).string());
    header.replace_extension(".hpp");
    metaFiles["require.header"].insert(header.string());
    metaFiles["require.library"].insert(internalName);
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
  Compiler compiler(package, work_dir, build_dir);
  compiler.parse();
  compiler.resolve();
  compiler.gen();
  delete package;
}

void bi::Driver::autogen() {
  if (newAutogen || newConfigure || newMake || newMeta
      || !exists(work_dir / "configure")
      || !exists(work_dir / "install-sh")) {
    std::stringstream cmd;

    cmd << (work_dir / "autogen.sh");
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
        buf << ", see " << (build_dir / "autogen.log").string()
            << " for details";
      }
      buf << '.';
      throw DriverException(buf.str());
    }
  }
}

void bi::Driver::configure() {
  if (newAutogen || newConfigure || newMake || newMeta
      || !exists(build_dir / "Makefile")) {
    /* working directory */
    std::stringstream cppflags, cxxflags, ldflags, options, cmd;

    /* compile and link flags */
    if (arch == "js") {
      //
    } else if (arch == "wasm") {
      cxxflags << " -s WASM=1";
    } else if (arch != "native") {
      throw DriverException("unknown architecture '" + arch
              + "'; valid values are 'native', 'js' and 'wasm'");
    }
    if (debug) {
      cppflags << " -D_GLIBCXX_DEBUG";
      cxxflags << " -O0 -g -fno-inline";
      ldflags << " -O0 -g -fno-inline";
    } else {
      cppflags << " -DNDEBUG";

      /*
       * -flto enables link-time code generation, which is used in favour
       * of explicitly inlining functions written in Birch. The gcc manpage
       * recommends passing the same optimisation options to the linker as
       * to the compiler when using this.
       */
      cxxflags << " -O3 -g -funroll-loops -flto";
      ldflags << " -O3 -g -funroll-loops -flto";
    }
    if (warnings) {
      cxxflags << " -Wall";
      ldflags << " -Wall";
    }
    cxxflags << " -Wno-overloaded-virtual";

    for (auto iter = include_dirs.begin(); iter != include_dirs.end();
        ++iter) {
      cppflags << " -I" << iter->string();
    }
    for (auto iter = lib_dirs.begin(); iter != lib_dirs.end(); ++iter) {
      ldflags << " -L" << iter->string();
    }
    for (auto iter = lib_dirs.begin(); iter != lib_dirs.end(); ++iter) {
      ldflags << " -Wl,-rpath," << iter->string();
    }

    /* configure options */
    if (!prefix.empty()) {
      options << " --prefix=" << absolute(prefix);
    }
    options << " INSTALL=\"install -p\"";
    options << " --config-cache";

    /* command */
    if (arch == "js" || arch == "wasm") {
      cmd << "emconfigure ";
    }
    cmd << (work_dir / "configure") << " " << options.str();
    if (!cppflags.str().empty()) {
      cmd << " CPPFLAGS='" << cppflags.str() << "'";
    }
    if (!cxxflags.str().empty()) {
      cmd << " CXXFLAGS='" << cxxflags.str() << "'";
    }
    if (!ldflags.str().empty()) {
      cmd << " LDFLAGS='" << ldflags.str() << "'";
    }
    if (verbose) {
      std::cerr << cmd.str() << std::endl;
    } else {
      cmd << " > configure.log 2>&1";
    }

    /* change into build dir */
    current_path(build_dir);

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
    current_path(work_dir);
  }
}

void bi::Driver::target(const std::string& cmd) {
  /* command */
  std::stringstream buf;
  if (arch == "js" || arch == "wasm") {
    buf << "emmake ";
  }
  buf << "make -j 4 " << cmd;

  /* handle output */
  std::string log = cmd + ".log";
  if (verbose) {
    std::cerr << buf.str() << std::endl;
  } else {
    buf << " > " << log << " 2>&1";
  }

  /* change into build dir */
  current_path(build_dir);

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
  current_path(work_dir);
}

void bi::Driver::readFiles(const boost::property_tree::ptree& meta,
    const std::string& key, bool checkExists) {
  auto files = meta.get_child_optional(key);
  if (files) {
    for (auto file : files.get()) {
      if (auto str = file.second.get_value_optional<std::string>()) {
        if (str) {
          fs::path filePath(str.get());
          std::string fileStr = filePath.string();
          if (checkExists && !exists(filePath)) {
            warn(fileStr + " in META.json does not exist.");
          }
          if (std::regex_search(fileStr, std::regex("\\s", std::regex_constants::ECMAScript))) {
            throw DriverException(std::string("file name ") + fileStr + " in META.json contains whitespace, which is not supported.");
          }
          auto inserted = allFiles.insert(filePath);
          if (!inserted.second) {
            warn(fileStr + " repeated in META.json.");
          }
          metaFiles[key].insert(filePath);
        }
      }
    }
  }
}
