/**
 * @file
 */
#include "Driver.hpp"

#include "bi/build/Compiler.hpp"
#include "bi/build/misc.hpp"
#include "bi/exception/DriverException.hpp"
#include "bi/io/md_ostream.hpp"

#include "boost/filesystem/fstream.hpp"
#include "boost/algorithm/string.hpp"

#include <iostream>
#include <regex>
#include <unordered_set>
#include <getopt.h>
#include <dlfcn.h>

using namespace boost::filesystem;

bi::Driver::Driver(int argc, char** argv) :
    work_dir(current_path()),
    build_dir(current_path() / "build"),
    prefix(""),
    std(true),
    warnings(true),
    debug(true),
    verbose(true),
    newAutogen(false),
    newConfigure(false),
    newMake(false),
    newManifest(false),
    isLocked(false) {
  enum {
    SHARE_DIR_ARG = 256,
    INCLUDE_DIR_ARG,
    LIB_DIR_ARG,
    PREFIX_ARG,
    ENABLE_STD_ARG,
    DISABLE_STD_ARG,
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
      { "prefix", required_argument, 0, PREFIX_ARG },
      { "enable-std", no_argument, 0, ENABLE_STD_ARG },
      { "disable-std", no_argument, 0, DISABLE_STD_ARG },
      { "enable-warnings", no_argument, 0, ENABLE_WARNINGS_ARG },
      { "disable-warnings", no_argument, 0, DISABLE_WARNINGS_ARG },
      { "enable-debug", no_argument, 0, ENABLE_DEBUG_ARG },
      { "disable-debug", no_argument, 0, DISABLE_DEBUG_ARG },
      { "enable-verbose", no_argument, 0, ENABLE_VERBOSE_ARG },
      { "disable-verbose", no_argument, 0, DISABLE_VERBOSE_ARG },
      { 0, 0, 0, 0 }
  };
  const char* short_options = "-";  // treats non-options as short option 1

  /* read project name */
  readName();
  if (projectName.compare("birch_standard") == 0) {
    /* by default, disable inclusion of the standard library when the project
     * is, itself,  the standard library (!) */
    std = false;
  }

  /* mutable copy of argv and argc */
  largv.insert(largv.begin(), argv, argv + argc);
  std::vector<char*> fargv;

  /* first position argument is program name */
  int i = 1;

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
    case PREFIX_ARG:
      prefix = optarg;
      break;
    case ENABLE_STD_ARG:
      std = true;
      break;
    case DISABLE_STD_ARG:
      std = false;
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
  std::string path;

  /* share dirs */
  if (BIRCH_SHARE_PATH) {
    std::stringstream birch_share_path(BIRCH_SHARE_PATH);
    while (std::getline(birch_share_path, path, ':')) {
      share_dirs.push_back(path);
    }
  }
  share_dirs.push_back("share");
#ifdef DATADIR
  share_dirs.push_back(STRINGIFY(DATADIR));
#endif
  share_dirs.push_back("/usr/local/share");
  share_dirs.push_back("/usr/share");

  /* include dirs */
  include_dirs.push_back(work_dir);
  include_dirs.push_back(build_dir);
  if (BIRCH_INCLUDE_PATH) {
    std::stringstream birch_include_path(BIRCH_INCLUDE_PATH);
    while (std::getline(birch_include_path, path, ':')) {
      include_dirs.push_back(path);
    }
  }
#ifdef INCLUDEDIR
  include_dirs.push_back(STRINGIFY(INCLUDEDIR));
#endif
  include_dirs.push_back("/usr/local/include");
  include_dirs.push_back("/usr/include");

  /* lib dirs */
  if (BIRCH_LIBRARY_PATH) {
    std::stringstream birch_library_path(BIRCH_LIBRARY_PATH);
    while (std::getline(birch_library_path, path, ':')) {
      lib_dirs.push_back(path);
    }
  }
  //lib_dirs.push_back("lib");
#ifdef LIBDIR
  lib_dirs.push_back(STRINGIFY(LIBDIR));
#endif
  lib_dirs.push_back("/usr/local/lib");
  lib_dirs.push_back("/usr/lib");
}

bi::Driver::~Driver() {
  unlock();
}

void bi::Driver::run(const std::string& prog) {
  /* dynamically load possible programs */
  typedef void prog_t(int argc, char** argv);

  void* handle;
  void* addr;
  char* msg;
  prog_t* fcn;

  path so = std::string("lib") + projectName;
#ifdef __APPLE__
  so.replace_extension(".dylib");
#else
  so.replace_extension(".so");
#endif
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
  setup();
  autogen();
  configure();
  target();
}

void bi::Driver::install() {
  setup();
  autogen();
  configure();
  target("install");
}

void bi::Driver::uninstall() {
  setup();
  autogen();
  configure();
  target("uninstall");
}

void bi::Driver::dist() {
  setup();
  autogen();
  configure();
  target("dist");
}

void bi::Driver::clean() {
  //setup();
  //autogen();
  //configure();
  //target("clean");

  remove_all(build_dir);
  remove_all("autom4te.cache");
  remove_all("m4");
  remove("aclocal.m4");
  remove("autogen.log");
  remove("autogen.sh");
  remove("common.am");
  remove("compile");
  remove("config.guess");
  remove("config.sub");
  remove("configure");
  remove("configure.ac");
  remove("depcomp");
  remove("install-sh");
  remove("ltmain.sh");
  remove("Makefile.am");
  remove("Makefile.in");
  remove("missing");
}

void bi::Driver::init() {
  create_directory("bi");
  path biPath("bi");
  copy_with_prompt(find(share_dirs, "gitignore"), ".gitignore");
  copy_with_prompt(find(share_dirs, "LICENSE"), "LICENSE");
  copy_with_prompt(find(share_dirs, "MANIFEST"), "MANIFEST");
  copy_with_prompt(find(share_dirs, "README.md"), "README.md");
}

void bi::Driver::check() {
  std::unordered_set<std::string> manifestFiles;

  /* check MANIFEST */
  if (!exists("MANIFEST")) {
    warn(
        "no MANIFEST file; create a MANIFEST file with a list of files, one per line, to be contained in the package.");
  } else {
    ifstream manifestStream("MANIFEST");
    std::string name;
    while (std::getline(manifestStream, name)) {
      boost::trim(name);
      manifestFiles.insert(name);
      if (!exists(name)) {
        warn(name + " file listed in MANIFEST file does not exist.");
      }
    }
  }

  /* check LICENSE */
  if (!exists("LICENSE")) {
    warn(
        "no LICENSE file; create a LICENSE file containing the distribution license (e.g. GPL or BSD) of the package.");
  } else if (manifestFiles.find("LICENSE") == manifestFiles.end()) {
    warn("LICENSE file is not listed in MANIFEST file.");
  }

  /* check README.md */
  if (!exists("README.md")) {
    warn(
        "no README.md file; create a README.md file documenting the package in Markdown format.");
  } else if (manifestFiles.find("README.md") == manifestFiles.end()) {
    warn("README.md file is not listed in MANIFEST file.");
  }

  /* check for files that might be missing from MANIFEST */
  std::unordered_set<std::string> interesting;

  interesting.insert(".bi");
  interesting.insert(".conf");
  interesting.insert(".sh");
  interesting.insert(".cpp");
  interesting.insert(".hpp");
  interesting.insert(".m");
  interesting.insert(".R");

  recursive_directory_iterator iter("."), end;
  while (iter != end) {
    auto path = remove_first(iter->path());
    if (path.string().compare("build") == 0) {
      iter.no_push();
    } else if (interesting.find(path.extension().string())
        != interesting.end()) {
      if (manifestFiles.find(path.string()) == manifestFiles.end()) {
        warn(
            std::string("is ") + path.string()
                + " missing from MANIFEST file?");
      }
    }
    ++iter;
  }

  if (is_directory("data")) {
    interesting.clear();
    interesting.insert(".nc");
    recursive_directory_iterator dataIter("data");
    while (dataIter != end) {
      auto path = dataIter->path();
      if (interesting.find(path.extension().string()) != interesting.end()) {
        if (manifestFiles.find(path.string()) == manifestFiles.end()) {
          warn(
              std::string("is ") + path.string()
                  + " missing from MANIFEST file?");
        }
      }
      ++dataIter;
    }
  }
}

void bi::Driver::docs() {
  current_path(work_dir);
  readManifest();

  /* parse all files */
  Compiler compiler(include_dirs, lib_dirs, std);
  for (auto file : biFiles) {
    compiler.queue(file.string());
  }
  compiler.parse();
  compiler.resolve();

  /* output everything, categorised by object type, and sorted */
  path path = "DOCS.md";
  ofstream stream(path);
  md_ostream output(stream);

  Package package(compiler.files);
  output << &package;
}

void bi::Driver::unlock() {
  if (isLocked) {
    lockFile.unlock();
  }
  isLocked = false;
}

void bi::Driver::readName() {
  projectName = "birch_untitled";
  ifstream metaStream("README.md");
  std::regex reg("^name:\\s*(\\w+)$");
  std::smatch match;
  std::string line;
  while (std::getline(metaStream, line)) {
    if (std::regex_match(line, match, reg)) {
      projectName = "birch_";
      projectName += match[1];
      boost::algorithm::to_lower(projectName);
      break;
    }
  }
}

void bi::Driver::readManifest() {
  if (exists("MANIFEST")) {
    ifstream manifestStream("MANIFEST");
    std::string name;
    while (std::getline(manifestStream, name)) {
      path file(name);
      if (exists(file)) {
        files.push_back(file);

        /* collate by file extension */
        if (file.extension().compare(".bi") == 0) {
          biFiles.push_back(file);
        } else if (file.extension().compare(".cpp") == 0) {
          cppFiles.push_back(file);
        } else if (file.extension().compare(".hpp") == 0) {
          hppFiles.push_back(file);
        } else if (file.extension().compare("") == 0
            || file.extension().compare(".md") == 0) {
          metaFiles.push_back(file);
        } else {
          otherFiles.push_back(file);
        }
      } else {
        std::stringstream buf;
        buf << file.string() << " in MANIFEST does not exist.";
        warn(buf.str());
      }
    }
  } else {
    throw DriverException("No MANIFEST file.");
  }
}

void bi::Driver::setup() {
  /* create build directory */
  if (!exists(build_dir)) {
    if (!create_directory(build_dir)) {
      std::stringstream buf;
      buf << "Could not create build directory " << build_dir << '.';
      throw DriverException(buf.str());
    }
    ofstream stream(build_dir / "lock");  // creates lock file
  }
  lock();

  /* copy built files into build directory */
  path biPath("birch");
  newAutogen = copy_if_newer(find(share_dirs, biPath / "autogen.sh"),
      work_dir / "autogen.sh");
  permissions(work_dir / "autogen.sh", add_perms | owner_exe);
  newConfigure = copy_if_newer(find(share_dirs, biPath / "configure.ac"),
      work_dir / "configure.ac");
  newMake = copy_if_newer(find(share_dirs, biPath / "common.am"),
      work_dir / "common.am");
  newManifest = !exists(work_dir / "Makefile.am")
      || last_write_time("MANIFEST")
          > last_write_time(work_dir / "Makefile.am");

  path m4_dir = work_dir / "m4";
  if (!exists(m4_dir)) {
    if (!create_directory(m4_dir)) {
      std::stringstream buf;
      buf << "Could not create m4 directory " << m4_dir << '.';
      throw DriverException(buf.str());
    }
  }
  copy_if_newer(find(share_dirs, biPath / "ax_cxx_compile_stdcxx.m4"),
      m4_dir / "ax_cxx_compile_stdcxx.m4");
  copy_if_newer(find(share_dirs, biPath / "ax_cxx_compile_stdcxx_11.m4"),
      m4_dir / "ax_cxx_compile_stdcxx_11.m4");
  copy_if_newer(find(share_dirs, biPath / "ax_cxx_compile_stdcxx_14.m4"),
      m4_dir / "ax_cxx_compile_stdcxx_14.m4");
  copy_if_newer(find(share_dirs, biPath / "precompile.hpp"),
      build_dir / "precompile.hpp");

  /* build list of source files */
  readManifest();

  /* create Makefile.am */
  if (newManifest) {
    ofstream makeStream(work_dir / "Makefile.am");
    makeStream << "include common.am\n\n";
    makeStream << "lib_LTLIBRARIES = lib" << projectName << ".la\n\n";

    /* *.cpp files */
    makeStream << "lib" << projectName << "_la_SOURCES = ";
    auto iter = cppFiles.begin();
    while (iter != cppFiles.end()) {
      if (iter != cppFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != cppFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* sources derived from *.bi files */
    makeStream << "nodist_lib" << projectName << "_la_SOURCES = ";
    iter = biFiles.begin();
    while (iter != biFiles.end()) {
      iter->replace_extension(".cpp");
      if (iter != biFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != biFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* headers to install and distribute */
    makeStream << "nobase_include_HEADERS = ";
    iter = hppFiles.begin();
    while (iter != hppFiles.end()) {
      if (iter != hppFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != hppFiles.end() || !biFiles.empty()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    iter = biFiles.begin();
    while (iter != biFiles.end()) {
      iter->replace_extension(".bi");
      if (iter != biFiles.begin() || !hppFiles.empty()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != biFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* headers to install but not distribute */
    makeStream << "nobase_nodist_include_HEADERS = ";
    iter = biFiles.begin();
    while (iter != biFiles.end()) {
      iter->replace_extension(".hpp");
      if (iter != biFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != biFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* built sources */
    makeStream << "BUILT_SOURCES += ";
    iter = biFiles.begin();
    while (iter != biFiles.end()) {
      iter->replace_extension(".cpp");
      if (iter != biFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      makeStream << " \\\n  ";
      iter->replace_extension(".hpp");
      makeStream << iter->string();
      if (++iter != biFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* clean files */
    makeStream << "CLEANFILES += ";
    iter = biFiles.begin();
    while (iter != biFiles.end()) {
      iter->replace_extension(".cpp");
      if (iter != biFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      makeStream << " \\\n  ";
      iter->replace_extension(".hpp");
      makeStream << iter->string();
      if (++iter != biFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* data files */
    makeStream << "dist_pkgdata_DATA = ";
    iter = otherFiles.begin();
    while (iter != otherFiles.end()) {
      if (iter != otherFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != otherFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    /* meta files */
    makeStream << "noinst_DATA = ";
    iter = metaFiles.begin();
    while (iter != metaFiles.end()) {
      if (iter != metaFiles.begin()) {
        makeStream << "  ";
      }
      makeStream << iter->string();
      if (++iter != metaFiles.end()) {
        makeStream << " \\";
      }
      makeStream << '\n';
    }
    makeStream << '\n';

    makeStream.close();
  }
}

void bi::Driver::autogen() {
  if (newAutogen || newConfigure || newMake || newManifest
      || !exists(work_dir / "configure")
      || !exists(work_dir / "install-sh")) {
    std::stringstream cmd;

    cmd << (path(".") / "autogen.sh").string();
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
          << ". Make sure autoconf, automake and libtool are installed.";
      if (!verbose) {
        buf << " See " << (build_dir / "autogen.log").string() << " for details.";
      }
      throw DriverException(buf.str());
    }
  }
}

void bi::Driver::configure() {
  if (newAutogen || newConfigure || newMake || newManifest
      || !exists(build_dir / "Makefile")) {
    /* working directory */
    std::stringstream cppflags, cxxflags, ldflags, options, cmd;

    /* compile and link flags */
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
      // ^ can also use -flto=n to use n threads internally
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
      options << " --prefix=" << absolute(prefix).string();
    }
    options << " --disable-static";
    options << " INSTALL=\"install -p\"";
    // ^ This can be problematic for headers, as while *.bi file may change,
    //   this may not change the *.hpp file, and so make keeps trying to
    //   rebuild it. The solution has been to append the entire *.bi file to
    //   the end of the generated *.hpp file to ensure that it always changes.
    //   see Compiler.
    options << " --config-cache";
    options << (std ? " --enable-std" : " --disable-std");

    /* command */
    cmd << (work_dir / "configure").string() << " " << options.str()
        << " CPPFLAGS='" << cppflags.str() << "' CXXFLAGS='" << cxxflags.str()
        << "' LDFLAGS='" << ldflags.str() << "'";
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
          << ". Make sure all dependencies are installed.";
      if (!verbose) {
        buf << "See "
            << (build_dir / "configure.log").string() << " and "
            << (build_dir / "config.log").string() << " for details.";
      }
      throw DriverException(buf.str());
    }

    /* change back to original working dir */
    current_path(work_dir);
  }
}

void bi::Driver::target(const std::string& cmd) {
  /* command */
  std::stringstream buf;
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
      buf << " failed to execute.";
    } else {
      buf << " died with signal " << ret << '.';
    }
    if (!verbose) {
      buf << " See " << (build_dir / log).string() << " for details.";
    }
    throw DriverException(buf.str());
  }

  /* change back to original working dir */
  current_path(work_dir);
}

void bi::Driver::lock() {
  boost::interprocess::file_lock lockFile1((build_dir / "lock").c_str());
  lockFile.swap(lockFile1);
  lockFile.lock();
  isLocked = true;
}
