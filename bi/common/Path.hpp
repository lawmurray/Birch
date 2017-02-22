/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"
#include "bi/common/Name.hpp"
#include "bi/primitive/shared_ptr.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
class Visitor;

/**
 * Path in import statement.
 *
 * @ingroup compiler_common
 */
class Path: public Located {
public:
  /**
   * Constructor.
   *
   * @param head First name in path.
   * @param tail Remaining path.
   * @param loc Location.
   */
  Path(shared_ptr<Name> head, Path* tail = nullptr, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Path();

  virtual void accept(Visitor* visitor) const;

  virtual bool operator==(const Path& o) const;
  bool operator!=(Path& o);

  /**
   * Path as file name.
   */
  std::string file() const;

  /**
   * Path as string.
   */
  std::string str() const;

  /**
   * First name in path.
   */
  shared_ptr<Name> head;

  /**
   * Remaining path.
   */
  unique_ptr<Path> tail;
};
}
