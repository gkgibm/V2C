; ─────────────────────────────────────────────────────────────────────────────
; V2C Tree-sitter Python query file
;
; These S-expression patterns are used by v2c.ast_engine.queries to locate
; and capture Python AST nodes for voice-driven structural edits.
;
; Reference: https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries
; ─────────────────────────────────────────────────────────────────────────────


; ─── Function definitions ────────────────────────────────────────────────────

(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (_)? @function.return_type
  body: (block) @function.body) @function.def

; Decorated functions
(decorated_definition
  (decorator) @function.decorator
  definition: (function_definition
    name: (identifier) @function.name
    parameters: (parameters) @function.params
    body: (block) @function.body)) @function.decorated


; ─── Class definitions ───────────────────────────────────────────────────────

(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def


; ─── Method definitions (inside a class) ─────────────────────────────────────

(class_definition
  name: (identifier) @method.class_name
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.params
      body: (block) @method.body) @method.def))


; ─── Import statements ───────────────────────────────────────────────────────

(import_statement
  name: (dotted_name) @import.module) @import.stmt

(import_from_statement
  module_name: (dotted_name) @import.from_module
  name: (dotted_name) @import.symbol) @import.from_stmt


; ─── Variable assignments (top-level) ────────────────────────────────────────

(module
  (expression_statement
    (assignment
      left: (identifier) @var.name
      right: (_) @var.value) @var.assign))


; ─── All identifiers (for context-aware refinement) ──────────────────────────

(identifier) @identifier
