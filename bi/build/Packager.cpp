/**
 * @file
 */
#include "Packager.hpp"

#include "bi/build/misc.hpp"
#include "bi/config.hpp"

#include "boost/filesystem/fstream.hpp"
#include "boost/algorithm/string.hpp"

#include <sstream>
#include <unordered_set>
#include <getopt.h>

bi::Packager::Packager(int argc, char** argv) :
    force(false),
    verbose(false) {
  using namespace boost::filesystem;

  enum {
    SHARE_DIR_ARG,
    FORCE_ARG,
    VERBOSE_ARG
  };

  int c, option_index;
  option long_options[] = {
      { "share-dir", required_argument, 0, SHARE_DIR_ARG },
      { "force", no_argument, 0, FORCE_ARG },
      { "verbose", no_argument, 0, VERBOSE_ARG },
      { 0, 0, 0, 0 } };
  const char* short_options = "";

  /* mutable copy of argv and argc */
  std::vector<char*> largv(argv, argv + argc);
  std::list<std::string> lbufs;

  /* handle positional arguments as they arrive; needed for program name and
   * config files */
  setenv("POSIXLY_CORRECT", "1", 1);

  /* remaining options */
  while (optind < largv.size()) {  // don't use c != -1, this indicates a config file
    c = getopt_long(largv.size(), largv.data(), short_options, long_options,
        &option_index);
    switch (c) {
    case SHARE_DIR_ARG:
      share_dirs.push_back(optarg);
      break;
    case FORCE_ARG:
      force = true;
      break;
    case VERBOSE_ARG:
      verbose = true;
      break;
    case -1:
      /* assume config file */
      char* name;
      if (*largv[optind] == '@') {
        // for compatibility with LibBi, allow config file name
        // to start with '@', but just remove it
        name = largv[optind] + 1;
      } else {
        name = largv[optind];
      }
      largv.erase(largv.begin() + optind, largv.begin() + optind + 1);

      /* read in config file */
      yyin = fopen(name, "r");
      if (yyin) {
        int i = 0;
        while (yylex()) {
          lbufs.push_back(yytext);
          largv.insert(largv.begin() + optind + i,
              const_cast<char*>(lbufs.back().c_str()));
          ++i;
        }
      } else {
        warn(
            (std::stringstream() << "could not open config file " << name << '.').str());
      }
      break;
    }
  }

  /* environment variables */
  char* BIRCH_SHARE_PATH = getenv("BIRCH_SHARE_PATH");
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
  share_dirs.push_back("/usr/local/share/birch");
  share_dirs.push_back("/usr/share/birch");
}

bi::Packager::~Packager() {
  //
}

void bi::Packager::create() {
  using namespace boost::filesystem;

  create_directory("bi");
  path biPath("bi");
  if (force) {
    copy_with_force(find(share_dirs, "gitignore"), ".gitignore");
    copy_with_force(find(share_dirs, "LICENSE"), "LICENSE");
    copy_with_force(find(share_dirs, "Makefile"), "Makefile");
    copy_with_force(find(share_dirs, "MANIFEST"), "MANIFEST");
    copy_with_force(find(share_dirs, "META.md"), "META.md");
    copy_with_force(find(share_dirs, "README.md"), "README.md");
    copy_with_force(find(share_dirs, "VERSION.md"), "VERSION.md");
  } else {
    copy_with_prompt(find(share_dirs, "gitignore"), ".gitignore");
    copy_with_prompt(find(share_dirs, "LICENSE"), "LICENSE");
    copy_with_prompt(find(share_dirs, "Makefile"), "Makefile");
    copy_with_prompt(find(share_dirs, "MANIFEST"), "MANIFEST");
    copy_with_prompt(find(share_dirs, "META.md"), "META.md");
    copy_with_prompt(find(share_dirs, "README.md"), "README.md");
    copy_with_prompt(find(share_dirs, "VERSION.md"), "VERSION.md");
  }
}

void bi::Packager::validate() {
  using namespace boost::filesystem;

  std::unordered_set<std::string> manifestFiles;

  /* check MANIFEST */
  if (!exists("MANIFEST")) {
    warn(
        (std::stringstream()
            << "no MANIFEST file; create a MANIFEST file with a list of files, one per line, to be contained in the package.").str());
  } else {
    ifstream manifestStream("MANIFEST");
    std::string name;
    while (std::getline(manifestStream, name)) {
      boost::trim(name);
      manifestFiles.insert(name);
      if (!exists(name)) {
        warn(
            (std::stringstream() << name
                << " file listed in MANIFEST file does not exist.").str());
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

  /* check META.md */
  if (!exists("META.md")) {
    warn(
        "no META.md file; create a META.md file documenting the package in Markdown format.");
  } else if (manifestFiles.find("META.md") == manifestFiles.end()) {
    warn("META.md file is not listed in MANIFEST file.");
  }

  /* check README.md */
  if (!exists("README.md")) {
    warn(
        "no README.md file; create a README.md file documenting the package in Markdown format.");
  } else if (manifestFiles.find("README.md") == manifestFiles.end()) {
    warn("README.md file is not listed in MANIFEST file.");
  }

  /* check VERSION.md */
  if (!exists("VERSION.md")) {
    warn(
        "no VERSION.md file; create a VERSION.md file documenting changes to the package in Markdown format.");
  } else if (manifestFiles.find("VERSION.md") == manifestFiles.end()) {
    warn("VERSION.md file is not listed in MANIFEST file.");
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
            (std::stringstream() << "is " << path.string()
                << " missing from MANIFEST file?").str());
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
              (std::stringstream() << "is " << path.string()
                  << " missing from MANIFEST file?").str());
        }
      }
      ++dataIter;
    }
  }
}

void bi::Packager::distribute() {

}
