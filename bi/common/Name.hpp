/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

#include <string>
#include <cstring>

namespace bi {
class Cloner;
class Modifier;
class Visitor;

/**
 * Name.
 */
class Name: public Located {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Name(const std::string& name, shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Name(const char* name = "", shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Name(const char name, shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   *
   * @param sigil Sigil.
   * @param name Name.
   * @param loc Location.
   */
  Name(const char sigil, const std::string& name, shared_ptr<Location> loc =
      nullptr);

  /**
   * Constructor.
   *
   * @param sigil Sigil.
   * @param name Name.
   * @param loc Location.
   */
  Name(const char sigil, const char* name = "", shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Name();

  /**
   * Get name as string.
   */
  const std::string& str() const;

  /**
   * Is this name non-empty?
   */
  operator bool() const;

  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(const Name& o) const;
  virtual bool operator==(const Name& o) const;
  bool operator<(const Name& o) const;
  bool operator>(const Name& o) const;
  bool operator>=(const Name& o) const;
  bool operator!=(const Name& o) const;

private:
  /**
   * Name.
   */
  std::string name;
};
}
