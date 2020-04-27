/**
 * @file
 */
#include "bi/io/cpp/CppResumeGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppResumeGenerator::CppResumeGenerator(const Class* currentClass,
    const Fiber* currentFiber, std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    currentClass(currentClass),
    currentFiber(currentFiber),
    paramIndex(0),
    localIndex(currentClass ? 1 : 0) {
  // ^ first local variable is self, for member fiber
}

void bi::CppResumeGenerator::visit(const Function* o) {
  auto fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);
  auto generic = o->typeParams->width() + o->params->width() > 0;
  auto params = requiresParam(o);
  auto locals = requiresLocal(o);
  bool first = true;

  if (currentClass && !header) {
    genTemplateParams(currentClass);
  }
  genSourceLine(o->loc);
  if (generic || params || locals) {
    start("template<");
  }
  for (auto typeParam : *o->typeParams) {
    first = false;
    middle("class " << typeParam << ',');
  }
  if (params) {
    if (!first) {
      middle(", ");
    }
    first = false;
    middle("class Param_");
  }
  if (locals) {
    if (!first) {
      middle(", ");
    }
    first = false;
    middle("class Local_");
  }
  if (generic || params || locals) {
    finish('>');
  }
  if (header) {
    genSourceLine(o->loc);
    start("struct ");
    genUniqueName(o);
    middle(" final : public libbirch::FiberState<");
    finish(fiberType->yieldType << ',' << fiberType->returnType << "> {");
    in();

    if (params) {
      genSourceLine(o->loc);
      line("Param_ param_;\n");
    }
    if (locals) {
      genSourceLine(o->loc);
      line("Local_ local_;\n");
    }

    genSourceLine(o->loc);
    start("");
    genUniqueName(o);
    middle('(');
    if (params) {
      middle("const Param_& param_");
    }
    if (locals) {
      if (params) {
        middle(", ");
      }
      middle("const Local_& local_");
    }
    middle(") ");
    if (params || locals) {
      middle(":");
    }
    if (params) {
      middle(" param_(param_)");
    }
    if (locals) {
      if (params) {
        middle(',');
      }
      middle(" local_(local_)");
    }
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");

    genSourceLine(o->loc);
    line("virtual " << fiberType << " query();");
    if (currentClass) {
      start("LIBBIRCH_MEMBER_FIBER(");
    } else {
      start("LIBBIRCH_FIBER(");
    }
    genUniqueName(o);
    middle(", libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << ">)");
    start("LIBBIRCH_MEMBERS(");
    if (params) {
      middle("param_");
    }
    if (locals) {
      if (params) {
        middle(", ");
      }
      middle("local_");
    }
    finish(')');
    out();
    line("};\n");
    genYieldMacro(o);
  } else {
    genSourceLine(o->loc);
    start(fiberType << " bi::");
    if (currentClass) {
      middle("type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    genUniqueName(o);
    if (generic || params || locals) {
      start('<');
    }
    first = true;
    for (auto typeParam : *o->typeParams) {
      if (!first) {
        middle(", ");
      }
      first = false;
      middle("class " << typeParam);
    }
    if (params) {
      if (!first) {
        middle(", ");
      }
      first = false;
      middle("Param_");
    }
    if (locals) {
      if (!first) {
        middle(", ");
      }
      first = false;
      middle("Local_");
    }
    if (generic || params || locals) {
      middle('>');
    }
    finish("::query() {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    genUnpackParam(o);
    *this << o->braces->strip();
    out();
    line("}");
  }
}

void bi::CppResumeGenerator::visit(const MemberFunction* o) {
  visit(dynamic_cast<const Function*>(o));
}

void bi::CppResumeGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("yield_");
  genUniqueName(o);
  finish('(' << o->single << ");");
}

void bi::CppResumeGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return " << currentFiber->returnType << '(');
  if (!o->single->isEmpty()) {
    middle(o->single);
  }
  finish(");");
}

void bi::CppResumeGenerator::visit(const LocalVariable* o) {
  auto name = getName(o->name->str(), o->number);
  if (o->has(RESUME)) {
    start("[[maybe_unused]] auto& " << name);
    middle("/* " << o->number << " */");
    finish(" = local_.template get<" << localIndex++ << ">();");
  } else {
    genTraceLine(o->loc);
    if (o->has(AUTO)) {
      start("auto " << name);
    } else {
      start(o->type << ' ' << name);
    }
    middle("/* " << o->number << " */");
    genInit(o);
    finish(';');
  }
}

void bi::CppResumeGenerator::visit(const NamedExpression* o) {
  if (o->isLocal()) {
    middle(getName(o->name->str(), o->number));
    middle("/* " << o->number << " */");
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
  } else {
    CppBaseGenerator::visit(o);
  }
}

void bi::CppResumeGenerator::genUniqueName(const Numbered* o) {
  if (currentClass) {
    middle(currentClass->name << '_');
  }
  middle(currentFiber->name << '_' << currentFiber->number << '_');
  middle(o->number << '_');
}

void bi::CppResumeGenerator::genPackType(const Function* o) {
  if (!currentFiber->typeParams->isEmpty() || requiresParam(o) ||
      requiresLocal(o)) {
    bool first = true;
    middle('<');
    if (!currentFiber->typeParams->isEmpty()) {
      middle(currentFiber->typeParams);
      first = false;
    }
    if (requiresParam(o)) {
      if (!first) {
        middle(',');
      }
      middle("decltype(");
      genPackParam(o);
      middle(')');
      first = false;
    }
    if (requiresLocal(o)) {
      if (!first) {
        middle(',');
      }
      middle("decltype(");
      genPackLocal(o);
      middle(')');
      first = false;
    }
    middle('>');
  }
}

void bi::CppResumeGenerator::genPackParam(const Function* o) {
  Gatherer<Parameter> params;
  o->accept(&params);

  middle("libbirch::make_tuple(");
  bool first = true;
  for (auto param : params) {
    if (!first) {
      middle(", ");
    }
    first = false;
    middle(getName(param->name->str(), param->number));
  }
  middle(')');
}

void bi::CppResumeGenerator::genPackLocal(const Function* o) {
  Gatherer<LocalVariable> locals([](auto o) { return o->has(RESUME); });
  o->accept(&locals);

  middle("libbirch::make_tuple(");
  bool first = true;
  if (currentClass) {
    first = false;
    middle("self_()");
  }
  for (auto local : locals) {
    if (!first) {
      middle(", ");
    }
    first = false;
    middle(getName(local->name->str(), local->number));
  }
  middle(')');
}

void bi::CppResumeGenerator::genUnpackParam(const Function* o) {
  Gatherer<Parameter> params;
  o->params->accept(&params);

  for (auto param : params) {
    genSourceLine(param->loc);
    start("const auto& " << getName(param->name->str(), param->number));
    finish(" = param_.template get<" << paramIndex++ << ">();");
  }
}

void bi::CppResumeGenerator::genYieldMacro(const Function* o) {
  auto params = requiresParam(o);
  auto locals = requiresLocal(o);

  start("#define yield_");
  genUniqueName(o);
  middle('(');
  if (!o->has(START)) {
    middle("...");
  }
  middle(") return " << currentFiber->returnType << '(');
  if (!o->has(START)) {
    middle("__VA_ARGS__");
  }
  if (!o->has(START)) {
    middle(", ");
  }
  middle("new ");
  genUniqueName(o);
  genPackType(o);
  middle('(');
  if (params) {
    genPackParam(o);
  }
  if (locals) {
    if (params) {
      middle(", ");
    }
    genPackLocal(o);
  }
  middle(')');
  finish(")\n");
}

bool bi::CppResumeGenerator::requiresParam(const Function* o) {
  Gatherer<Parameter> params;
  o->accept(&params);
  return params.size() > 0;
}

bool bi::CppResumeGenerator::requiresLocal(const Function* o) {
  Gatherer<LocalVariable> locals([](auto o) { return o->has(RESUME); });
  o->accept(&locals);
  return currentClass || locals.size() > 0;
}

std::string bi::CppResumeGenerator::getName(const std::string& name,
    const int number) {
  std::stringstream buf;
  std::string result;
  auto iter = names.find(number);
  if (iter == names.end()) {
    auto count = counts.find(name);
    if (count == counts.end()) {
      buf << internalise(name);
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << internalise(name) << '_' << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}

std::string bi::CppResumeGenerator::getIndex(const Statement* o) {
  auto index = dynamic_cast<const LocalVariable*>(o);
  assert(index);
  return getName(index->name->str(), index->number);
}
