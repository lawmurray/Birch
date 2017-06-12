/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/statement/FuncParameter.hpp"
#include "bi/statement/BinaryParameter.hpp"
#include "bi/statement/UnaryParameter.hpp"
#include "bi/statement/AssignmentParameter.hpp"
#include "bi/type/TypeParameter.hpp"
#include "bi/statement/ProgParameter.hpp"

template<class Target>
bi::Reference<Target>::Reference(const Target* target) :
    target(target) {
  //
}

template<class Target>
bi::Reference<Target>::~Reference() {
  //
}

/*
 * Explicit instantiations.
 */
template class bi::Reference<bi::VarParameter>;
template class bi::Reference<bi::FuncParameter>;
template class bi::Reference<bi::BinaryParameter>;
template class bi::Reference<bi::UnaryParameter>;
template class bi::Reference<bi::AssignmentParameter>;
template class bi::Reference<bi::TypeParameter>;
template class bi::Reference<bi::ProgParameter>;
