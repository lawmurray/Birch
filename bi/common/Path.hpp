/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"
#include "bi/common/Name.hpp"

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
  Path(Name* head, Path* tail = nullptr, Location* loc =
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
  Name* head;

  /**
   * Remaining path.
   */
  Path* tail;
};
}
