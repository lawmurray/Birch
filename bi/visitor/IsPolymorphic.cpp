/**
 * @file
 */
#include "bi/visitor/IsPolymorphic.hpp"

bi::IsPolymorphic::IsPolymorphic() : result(false) {
  //
}

bi::IsPolymorphic::~IsPolymorphic() {
  //
}

void bi::IsPolymorphic::visit(const FuncDeclaration* o) {
  result = result || o->param->isVirtual();
}
