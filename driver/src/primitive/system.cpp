/**
 * @file
 */
#include "src/primitive/system.hpp"

#include "src/common/Location.hpp"
#include "src/statement/File.hpp"
#include "src/exception/DriverException.hpp"

void birch::warn(const std::string& msg) {
  std::cerr << "warning: " << msg << std::endl;
}

void birch::warn(const std::string& msg, Location* loc) {
  if (loc->file) {
    std::cerr << loc->file->path;
    std::cerr << ':' << loc->firstLine;
    std::cerr << ':' << loc->firstCol;
    std::cerr << ": ";
  }
  std::cerr << "warning: " << msg << std::endl;
}

fs::path birch::find(const std::list<fs::path>& paths, const fs::path& path) {
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

std::list<fs::path> birch::glob(const std::string& pattern) {
  std::list<fs::path> results;
  glob_t matches;
  int rescode = glob(pattern.c_str(), 0, 0, &matches);
  if (rescode == 0) {
    for (int i = 0; i < (int)matches.gl_pathc; ++i) {
      results.push_back(matches.gl_pathv[i]);
    }
  }
  globfree(&matches);
  return results;
}

void birch::copy_file_writeable(fs::path src, fs::path dst) {
  using namespace fs;

  copy_file(src, dst);
  permissions(dst, status(dst).permissions()|perms::owner_write);
}

bool birch::copy_if_newer(fs::path src, fs::path dst) {
  using namespace fs;

  /* copy_file(src, dst, copy_option::overwrite_if_exists) seems problematic,
   * workaround... */
  bool result = false;
  if (!exists(dst)) {
    copy_file_writeable(src, dst);
    result = true;
  } else if (last_write_time(src) > last_write_time(dst)) {
    remove(dst);
    copy_file_writeable(src, dst);
    result = true;
  }
  return result;
}

bool birch::copy_with_prompt(fs::path src, fs::path dst) {
  using namespace fs;

  bool result = false;
  std::string ans;

  if (exists(dst)) {
    std::cout << dst.string() << " already exists, overwrite? [y/N] ";
    std::getline(std::cin, ans);
    if (ans.length() > 0 && (ans[0] == 'y' || ans[0] == 'Y')) {
      remove(dst);
      copy_file_writeable(src, dst);
      result = true;
    }
  } else {
    copy_file_writeable(src, dst);
    result = true;
  }
  return result;
}

void birch::copy_with_force(fs::path src, fs::path dst) {
  if (fs::exists(dst)) {
    fs::remove(dst);
    copy_file_writeable(src, dst);
  } else {
    copy_file_writeable(src, dst);
  }
}

fs::path birch::remove_first(const fs::path& path) {
  auto parent = path.parent_path();
  if (parent == path || parent.string().compare(".") == 0) {
    return fs::path() / path.filename();
  } else {
    return remove_first(path.parent_path()) / path.filename();
  }
}

fs::path birch::remove_common_prefix(const fs::path& base, const fs::path& path) {
  auto iter1 = base.begin();
  auto end1 = base.end();
  auto iter2 = path.begin();
  auto end2 = path.end();
  while (iter1 != end1 && iter2 != end2 && *iter1 == *iter2) {
    ++iter1;
    ++iter2;
  }
  if (iter2 != end2) {
    auto result = *iter2;
    ++iter2;
    while (iter2 != end2) {
      result /= *iter2;
      ++iter2;
    }
    return result;
  } else {
    return fs::path();
  }
}

std::string birch::read_all(const fs::path& path) {
  fs_stream::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

void birch::write_all(const fs::path& path, const std::string& contents) {
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }
  fs_stream::ofstream out(path);
  if (out.fail()) {
    std::stringstream buf;
    buf << "Could not open " << path.string() << " for writing.";
    throw DriverException(buf.str());
  }
  std::stringstream buf(contents);
  out << buf.rdbuf();
}

bool birch::write_all_if_different(const fs::path& path,
    const std::string& contents) {
  if (fs::exists(path)) {
    std::string old = read_all(path);
    if (contents != old) {
      write_all(path, contents);
      return true;
    }
  } else {
    write_all(path, contents);
    return true;
  }
  return false;
}

void birch::replace_tag(const fs::path& path, const std::string& tag,
    const std::string& value) {
  auto contents = read_all(path);
  contents = std::regex_replace(contents, std::regex(tag), value);
  fs_stream::ofstream stream(path);
  if (stream.fail()) {
    std::stringstream buf;
    buf << "Could not open " << path.string() << " for writing.";
    throw DriverException(buf.str());
  }
  stream << contents;
}
