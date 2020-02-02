/**
 * @file
 */
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppMemberFiberGenerator::CppMemberFiberGenerator(const Class* theClass,
    std::ostream& base, const int level, const bool header) :
    CppFiberGenerator(base, level, header),
    theClass(theClass) {
  //
}

void bi::CppMemberFiberGenerator::visit(const MemberFiber* o) {
  fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);

  /* generate a unique name (within this scope) for the state of the fiber */
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->params;
  std::string baseName = internalise(o->name->str()) + encode32(base.str());
  std::string stateName = baseName + "_FiberState_";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for state */
  if (header) {
    start("class " << stateName << " final : ");
    finish("public libbirch::FiberState<" << fiberType->yieldType << "> {");
    line("public:");
    in();
    line("using class_type_ = " << stateName << ';');
    line("using super_type_ = libbirch::FiberState<" << fiberType->yieldType << ">;\n");
    start("libbirch::LazySharedPtr<bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    finish("> self;");
    for (auto o : params) {
      line(o->type << ' ' << o->name << ';');
    }
    for (auto o : locals) {
      line(o->type << ' ' << getName(o->name->str(), o->number) << ';');
    }
  }

  /* constructor */
  if (header) {
    start(stateName << "(const libbirch::LazySharedPtr<");
    middle(theClass->name);
    genTemplateArgs(theClass);
    middle(">& self");
    if (!o->params->isEmpty()) {
      middle(", " << o->params);
    }
    finish(");");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    middle(stateName << "(const libbirch::LazySharedPtr<" << theClass->name);
    genTemplateArgs(theClass);
    middle(">& self");
    if (!o->params->isEmpty()) {
      middle(", " << o->params);
    }
    finish(") :");
    in();
    in();
    genSourceLine(o->loc);
    start("super_type_(" << (yields.size() + 1) << ')');
    finish(',');
    genSourceLine(o->loc);
    start("self(self)");
    for (auto param : params) {
      finish(',');
      genSourceLine(param->loc);
      start(param->name << '(' << param->name << ')');
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* deep copy constructor */
  if (header) {
    line(stateName << "(libbirch::Label* label, const " << stateName << "& o);");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    middle(stateName << "(libbirch::Label* label, const " << stateName << "& o)");
    finish(" :");
    in();
    in();
    genSourceLine(o->loc);
    start("super_type_(label, o)");
    finish(',');
    genSourceLine(o->loc);
    start("self(label, o.self)");
    for (auto o : params) {
      finish(',');
      genSourceLine(o->loc);
      start(o->name << "(libbirch::clone(label, o." << o->name << "))");
    }
    for (auto o : locals) {
      auto name = getName(o->name->str(), o->number);
      finish(',');
      genSourceLine(o->loc);
      start(o->name << "(libbirch::clone(label, o." << o->name << "))");
    }
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }

  /* copy constructor, destructor, assignment operator */
  if (header) {
    line("virtual ~" << stateName << "() = default;");
    line(stateName << "(const " << stateName << "&) = delete;");
    line(stateName << "& operator=(const " << stateName << "&) = delete;");
  }

  /* clone function */
  if (header) {
    line("virtual class_type_* clone_(libbirch::Label* label) const;");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("typename bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::class_type_* bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("clone_(libbirch::Label* label) const {");
    in();
    genSourceLine(o->loc);
    line("return new class_type_(label, *this);");
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* name function */
  if (header) {
    line("virtual const char* getClassName() const;");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("const char* bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("getClassName() const {");
    in();
    genSourceLine(o->loc);
    line("return \"" << stateName << "\";");
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* freeze function */
  if (header) {
    line("virtual void doFreeze_();");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("void bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("doFreeze_() {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doFreeze_();");
    genSourceLine(o->loc);
    line("freeze(self);");
    for (auto o : params) {
      genSourceLine(o->loc);
      line("freeze(" << o->name << ");");
    }
    for (auto o : locals) {
      genSourceLine(o->loc);
      line("freeze(" << getName(o->name->str(), o->number) << ");");
    }
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* thaw function */
  if (header) {
    line("virtual void doThaw_(libbirch::Label* label_);");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("void bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("doThaw_(libbirch::Label* label_) {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doThaw_(label_);");
    genSourceLine(o->loc);
    line("thaw(self, label_);");
    for (auto o : params) {
      genSourceLine(o->loc);
      line("thaw(" << o->name << ", label_);");
    }
    for (auto o : locals) {
      genSourceLine(o->loc);
      line("thaw(" << getName(o->name->str(), o->number) << ", label_);");
    }
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* finish function */
  if (header) {
    line("virtual void doFinish_();");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("void bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("doFinish_() {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doFinish_();");
    genSourceLine(o->loc);
    line("finish(self);");
    for (auto o : params) {
      genSourceLine(o->loc);
      line("finish(" << o->name << ");");
    }
    for (auto o : locals) {
      genSourceLine(o->loc);
      line("finish(" << getName(o->name->str(), o->number) << ");");
    }
    genSourceLine(o->loc);
    out();
    line("}");
  }

  /* query function */
  if (header) {
    line("virtual bool query();");
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle("void bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish("query() {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    line("libbirch_declare_local_");
    genSourceLine(o->loc);
    line("switch (local->point_) {");
    in();
    for (int s = 0; s <= yields.size(); ++s) {
      genSourceLine(o->loc);
      line("case " << s << ": goto POINT" << s << "_;");
    }
    genSourceLine(o->loc);
    line("default: goto END_;");
    genSourceLine(o->loc);
    out();
    line('}');
    genSourceLine(o->loc);
    line("POINT0_:");
    ++point;

    *this << o->braces->strip();

    genSourceLine(o->loc);
    line("END_:");
    genSourceLine(o->loc);
    line("local->point_ = " << (yields.size() + 1) << ';');
    genSourceLine(o->loc);
    line("return false;");
    genSourceLine(o->loc);
    out();
    finish("}\n");
  }
  if (header) {
    out();
    line("};\n");
  }

  /* initialisation function */
  if (header) {
    start("virtual " << fiberType << ' ' << o->name << '(' << o->params << ')');
    if (o->has(FINAL)) {
      middle(" final");
    } else if (o->has(ABSTRACT)) {
      middle(" = 0");
    }
    finish(';');
  } else {
    genSourceLine(o->loc);
    genTemplateParams(theClass);
    genSourceLine(o->loc);
    middle(fiberType << " bi::type::" << theClass->name);
    genTemplateArgs(theClass);
    middle("::" << stateName << "::");
    finish(o->name << '(' << o->params << ") {");
    in();
    genSourceLine(o->loc);
    line("libbirch_declare_self_");
    genSourceLine(o->loc);
    start("return libbirch::make_fiber<" << stateName << ">(self");
    for (auto param: params) {
      middle(", " << param->name);
    }
    finish(");");
    genSourceLine(o->loc);
    out();
    finish("}\n");
  }
}

void bi::CppMemberFiberGenerator::visit(const This* o) {
  middle("local->self");
}

void bi::CppMemberFiberGenerator::visit(const Super* o) {
  middle("local->self");
}
