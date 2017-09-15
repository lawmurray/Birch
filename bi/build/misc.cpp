/**
 * @file
 */
#include "bi/build/misc.hpp"

#include "boost/filesystem/fstream.hpp"

#include <iostream>
#include <sstream>

void bi::warn(const std::string& msg) {
  std::cerr << "warning: " << msg << std::endl;
}

boost::filesystem::path bi::find(
    const std::list<boost::filesystem::path>& paths,
    const boost::filesystem::path path) {
  if (exists(path)) {
    return path;
  } else {
    auto iter = paths.begin();
    while (iter != paths.end() && !exists(*iter / path)) {
      ++iter;
    }
    if (iter == paths.end()) {
      throw FileNotFoundException(path.string().c_str());
    } else {
      return *iter / path;
    }
  }
}

bool bi::copy_if_newer(boost::filesystem::path src,
    boost::filesystem::path dst) {
  using namespace boost::filesystem;

  /* copy_file(src, dst, copy_option::overwrite_if_exists) seems problematic,
   * workaround... */
  bool result = false;
  if (!exists(dst)) {
    copy(src, dst);
    result = true;
  } else if (last_write_time(src) > last_write_time(dst)) {
    remove(dst);
    copy(src, dst);
    result = true;
  }
  return result;
}

bool bi::copy_with_prompt(boost::filesystem::path src,
    boost::filesystem::path dst) {
  using namespace boost::filesystem;

  bool result = false;
  std::string ans;

  if (exists(dst)) {
    std::cout << dst.string() << " already exists, overwrite? [y/N] ";
    std::getline(std::cin, ans);
    if (ans.length() > 0 && (ans[0] == 'y' || ans[0] == 'Y')) {
      remove(dst);
      copy(src, dst);
      result = true;
    }
  } else {
    copy(src, dst);
    result = true;
  }
  return result;
}

void bi::copy_with_force(boost::filesystem::path src,
    boost::filesystem::path dst) {
  using namespace boost::filesystem;

  if (exists(dst)) {
    remove(dst);
    copy(src, dst);
  } else {
    copy(src, dst);
  }
}

boost::filesystem::path bi::remove_first(
    const boost::filesystem::path& path) {
  if (path.parent_path().string().compare(".") == 0) {
    return boost::filesystem::path() / path.filename();
  } else {
    return remove_first(path.parent_path()) / path.filename();
  }
}

std::string bi::read_all(const boost::filesystem::path& path) {
  boost::filesystem::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

void bi::write_all(const boost::filesystem::path& path,
    const std::string& contents) {
  boost::filesystem::ofstream out(path);
  std::stringstream buf(contents);
  out << buf.rdbuf();
}
