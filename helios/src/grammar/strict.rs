/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Strict function tool schema admission, syntax-aware grammar composition, and
//! byte-grammar validation of completed arguments.

use super::{
    GrammarCompiler, MAX_GRAMMAR_BYTES, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, OutputEnvelope,
    RequestGrammar, TextGrammar, validate_lark, validate_schema_nodes,
};
use crate::{AdmissionError, Error, Result, model::contract::ToolParserKind};
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::{Map, Value};
use std::collections::HashSet;

pub(crate) struct StrictToolSpec<'a> {
    pub name: String,
    pub strict_schema: Option<&'a Value>,
}

pub(crate) struct ToolGrammarPlan<'a> {
    pub tools: Vec<StrictToolSpec<'a>>,
    pub syntax: Option<(ToolParserKind, usize, String)>,
    pub dsml_token: Option<(u32, String)>,
    pub allow_undeclared: bool,
    pub require_call: bool,
    pub single_call: bool,
}

pub(crate) enum AnswerConstraint<'a> {
    JsonObject,
    Json(&'a Value),
    Lark(&'a str),
}

fn grammar_err(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::Grammar, message.into())
}

fn valid_wire_ident(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && !name.chars().any(|c| {
            c.is_control()
                || c.is_whitespace()
                || matches!(c, '"' | '\'' | '<' | '>' | '&' | '\\' | '\u{ff5c}')
        })
}

pub(crate) fn validate_strict_tool_schema(
    compiler: &GrammarCompiler,
    tool_name: &str,
    schema: &Value,
) -> Result<TextGrammar> {
    if !valid_wire_ident(tool_name) {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} name contains characters unsupported by tool wire framing"
        )));
    }
    if schema.to_string().len() > MAX_GRAMMAR_BYTES {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} JSON schema exceeds {MAX_GRAMMAR_BYTES} bytes"
        )));
    }
    let root = schema.as_object().ok_or_else(|| {
        grammar_err(format!(
            "strict tool {tool_name:?} parameters schema $ must be a JSON object"
        ))
    })?;
    if root.get("type") != Some(&Value::String("object".into())) {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} parameters schema $ must declare \"type\": \"object\""
        )));
    }
    for forbidden in ["anyOf", "oneOf", "allOf", "not", "$ref"] {
        if root.contains_key(forbidden) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} parameters schema $ cannot use root {forbidden}"
            )));
        }
    }

    let mut nodes = 0usize;
    let mut seen_patterns = HashSet::new();
    let mut validated_refs = HashSet::new();
    validated_refs.insert("#".to_owned());

    validate_strict_node(
        compiler,
        tool_name,
        root,
        schema,
        "$",
        0,
        &mut nodes,
        &mut HashSet::new(),
        &mut validated_refs,
        &mut seen_patterns,
    )?;

    for defs_key in ["$defs", "definitions"] {
        if let Some(defs_val) = root.get(defs_key) {
            let defs_obj = defs_val.as_object().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema $/{defs_key} must be a JSON object"
                ))
            })?;
            for (def_name, def_schema) in defs_obj {
                if def_name.is_empty() {
                    return Err(grammar_err(format!(
                        "strict tool {tool_name:?} schema $/{defs_key}/{def_name:?} has an invalid definition name"
                    )));
                }
                let def_obj = def_schema.as_object().ok_or_else(|| {
                    grammar_err(format!(
                        "strict tool {tool_name:?} schema $/{defs_key}/{def_name} must be a JSON object"
                    ))
                })?;
                let escaped = def_name.replace('~', "~0").replace('/', "~1");
                let canonical = format!("#/{defs_key}/{escaped}");
                let mut unguarded = HashSet::new();
                unguarded.insert(canonical.clone());
                validated_refs.insert(canonical);
                validate_strict_node(
                    compiler,
                    tool_name,
                    def_obj,
                    schema,
                    &format!("$/{defs_key}/{def_name}"),
                    1,
                    &mut nodes,
                    &mut unguarded,
                    &mut validated_refs,
                    &mut seen_patterns,
                )?;
            }
        }
    }

    check_unguarded_ref_cycles(tool_name, schema)?;
    compiler.compile_json_schema_text(schema)
}

const SUPPORTED_KEYS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "$ref",
    "$defs",
    "definitions",
    "description",
    "title",
    "default",
    "$schema",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
];

fn unsupported_strict_literal(value: &Value) -> bool {
    const LIMIT: u64 = 1 << 53;

    match value {
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                u >= LIMIT
            } else if let Some(i) = n.as_i64() {
                i.unsigned_abs() >= LIMIT
            } else {
                n.as_f64()
                    .is_some_and(|f| !f.is_finite() || f.abs() >= LIMIT as f64)
            }
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_strict_node(
    compiler: &GrammarCompiler,
    tool_name: &str,
    obj: &Map<String, Value>,
    root_schema: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
    unguarded_refs: &mut HashSet<String>,
    validated_refs: &mut HashSet<String>,
    seen_patterns: &mut HashSet<String>,
) -> Result<()> {
    *nodes += 1;
    if depth > MAX_SCHEMA_DEPTH || *nodes > MAX_SCHEMA_NODES {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} schema exceeds depth or node limits at {path}"
        )));
    }
    for key in obj.keys() {
        if !SUPPORTED_KEYS.contains(&key.as_str()) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema uses unsupported keyword {key:?} at {path}"
            )));
        }
    }
    if depth > 0 && (obj.contains_key("$defs") || obj.contains_key("definitions")) {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} schema {path} may only declare $defs/definitions at the root"
        )));
    }

    if let Some(ref_val) = obj.get("$ref") {
        let target = ref_val.as_str().ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/$ref must be a string"
            ))
        })?;
        for k in obj.keys() {
            if !matches!(k.as_str(), "$ref" | "description" | "title" | "default") {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path} cannot combine $ref with {k:?}"
                )));
            }
        }
        let resolved = resolve_ref_value(root_schema, target).ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/$ref {target:?} must resolve to #, #/$defs/<name>, or #/definitions/<name>"
            ))
        })?;
        let target_obj = resolved.as_object().ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/$ref target {target:?} must be an object"
            ))
        })?;
        if !unguarded_refs.insert(target.to_owned()) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema contains cyclic unguarded $ref {target:?} at {path}"
            )));
        }
        if validated_refs.insert(target.to_owned()) {
            validate_strict_node(
                compiler,
                tool_name,
                target_obj,
                root_schema,
                target,
                depth + 1,
                nodes,
                unguarded_refs,
                validated_refs,
                seen_patterns,
            )?;
        }
        unguarded_refs.remove(target);
        return Ok(());
    }

    if let Some(any_of) = obj.get("anyOf") {
        let branches = any_of.as_array().filter(|b| !b.is_empty()).ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/anyOf must be a nonempty array"
            ))
        })?;
        for k in obj.keys() {
            if !matches!(
                k.as_str(),
                "anyOf" | "description" | "title" | "default" | "$defs" | "definitions" | "$schema"
            ) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path} cannot combine anyOf with {k:?}"
                )));
            }
        }
        for (i, branch) in branches.iter().enumerate() {
            let b_obj = branch.as_object().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/anyOf/{i} must be a JSON object"
                ))
            })?;
            validate_strict_node(
                compiler,
                tool_name,
                b_obj,
                root_schema,
                &format!("{path}/anyOf/{i}"),
                depth + 1,
                nodes,
                unguarded_refs,
                validated_refs,
                seen_patterns,
            )?;
        }
        return Ok(());
    }

    if obj.contains_key("enum") || obj.contains_key("const") {
        let literal = if obj.contains_key("const") {
            "const"
        } else {
            "enum"
        };
        if let Some(key) = obj.keys().find(|k| {
            k.as_str() != literal
                && !matches!(k.as_str(), "type" | "description" | "title" | "default")
        }) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path} cannot combine {literal} with {key}"
            )));
        }
        let kinds = declared_types(tool_name, obj, path)?;
        if !kinds.is_empty()
            && obj
                .get("const")
                .into_iter()
                .chain(
                    obj.get("enum")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten(),
                )
                .any(|v| !kinds.iter().any(|kind| value_matches_kind(v, kind)))
        {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path} contains a literal outside its declared type"
            )));
        }
    }
    if let Some(c) = obj.get("const") {
        if c.is_object() || c.is_array() {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/const must be a JSON scalar"
            )));
        }
        if unsupported_strict_literal(c) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/const contains a number outside the exact 53-bit integer range"
            )));
        }
        return Ok(());
    }

    if let Some(en) = obj.get("enum") {
        let vals = en.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/enum must be a nonempty array"
            ))
        })?;
        for v in vals {
            if v.is_object() || v.is_array() {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/enum entries must be JSON scalars"
                )));
            }
            if unsupported_strict_literal(v) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/enum contains a number outside the exact 53-bit integer range"
                )));
            }
        }
        return Ok(());
    }

    let kinds = declared_types(tool_name, obj, path)?;
    if kinds.is_empty() {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} schema {path} must specify type, anyOf, enum, const, or $ref"
        )));
    }

    if kinds.contains(&"object") {
        if obj.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} object schema at {path} must set \"additionalProperties\": false"
            )));
        }
        let empty_props = Map::new();
        let props = match obj.get("properties") {
            Some(v) => v.as_object().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/properties must be an object"
                ))
            })?,
            None => &empty_props,
        };
        let req_list = match obj.get("required") {
            Some(v) => v.as_array().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/required must be an array"
                ))
            })?,
            None if props.is_empty() => &Vec::new(),
            None => {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} object schema at {path} must list all properties in \"required\""
                )));
            }
        };
        let mut req_seen = HashSet::with_capacity(req_list.len());
        for entry in req_list {
            let name = entry.as_str().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/required entries must be strings"
                ))
            })?;
            if !req_seen.insert(name) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/required duplicates {name:?}"
                )));
            }
            if !props.contains_key(name) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/required references undeclared property {name:?}"
                )));
            }
        }
        if req_seen.len() != props.len() {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} strict object schema at {path} must require every declared property"
            )));
        }
        for (prop_name, prop_schema) in props {
            if prop_name.contains(['<', '>']) || prop_name.chars().any(char::is_whitespace) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} property {prop_name:?} at {path} contains a delimiter that function-tag syntax cannot carry"
                )));
            }
            if depth == 0 && !valid_wire_ident(prop_name) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} top-level property {prop_name:?} contains unsupported delimiter characters"
                )));
            }
            let prop_obj = prop_schema.as_object().ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/properties/{prop_name} must be an object"
                ))
            })?;
            // Entering an object property guards recursive references.
            let mut child_unguarded = HashSet::new();
            validate_strict_node(
                compiler,
                tool_name,
                prop_obj,
                root_schema,
                &format!("{path}/properties/{prop_name}"),
                depth + 1,
                nodes,
                &mut child_unguarded,
                validated_refs,
                seen_patterns,
            )?;
        }
    } else if obj.contains_key("properties")
        || obj.contains_key("required")
        || obj.contains_key("additionalProperties")
    {
        return Err(grammar_err(format!(
            "strict tool {tool_name:?} non-object schema at {path} cannot declare object keywords"
        )));
    }

    if kinds.contains(&"array") {
        let min_items = opt_u64(tool_name, obj, "minItems", path)?;
        let max_items = opt_u64(tool_name, obj, "maxItems", path)?;
        if let (Some(mn), Some(mx)) = (min_items, max_items)
            && mn > mx
        {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path} has minItems > maxItems"
            )));
        }
        let has_prefix = if let Some(prefix_val) = obj.get("prefixItems") {
            let arr = prefix_val
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| {
                    grammar_err(format!(
                        "strict tool {tool_name:?} schema {path}/prefixItems must be a nonempty array"
                    ))
                })?;
            for (i, elem) in arr.iter().enumerate() {
                let elem_obj = elem.as_object().ok_or_else(|| {
                    grammar_err(format!(
                        "strict tool {tool_name:?} schema {path}/prefixItems/{i} must be an object"
                    ))
                })?;
                let mut child_unguarded = HashSet::new();
                validate_strict_node(
                    compiler,
                    tool_name,
                    elem_obj,
                    root_schema,
                    &format!("{path}/prefixItems/{i}"),
                    depth + 1,
                    nodes,
                    &mut child_unguarded,
                    validated_refs,
                    seen_patterns,
                )?;
            }
            true
        } else {
            false
        };
        match obj.get("items") {
            Some(Value::Bool(false)) if has_prefix => {}
            Some(item_val) => {
                let item_obj = item_val.as_object().ok_or_else(|| {
                    grammar_err(format!(
                        "strict tool {tool_name:?} schema {path}/items must be a schema object"
                    ))
                })?;
                let mut child_unguarded = HashSet::new();
                validate_strict_node(
                    compiler,
                    tool_name,
                    item_obj,
                    root_schema,
                    &format!("{path}/items"),
                    depth + 1,
                    nodes,
                    &mut child_unguarded,
                    validated_refs,
                    seen_patterns,
                )?;
            }
            None if has_prefix
                && max_items.is_some_and(|limit| {
                    limit <= obj["prefixItems"].as_array().unwrap().len() as u64
                }) => {}
            None => {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} array schema at {path} must constrain every element with items or a bounded tuple"
                )));
            }
        }
    }

    if kinds.contains(&"string") {
        let min_len = opt_u64(tool_name, obj, "minLength", path)?;
        let max_len = opt_u64(tool_name, obj, "maxLength", path)?;
        if let (Some(mn), Some(mx)) = (min_len, max_len)
            && mn > mx
        {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path} has minLength > maxLength"
            )));
        }
        if let Some(pat_val) = obj.get("pattern") {
            let pat = pat_val.as_str().filter(|p| !p.is_empty()).ok_or_else(|| {
                grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/pattern must be a nonempty string"
                ))
            })?;
            if !seen_patterns.contains(pat) {
                compiler.compile_text("regex", &unanchored_regex(pat))?;
                seen_patterns.insert(pat.to_owned());
            }
        }
    }

    if kinds.contains(&"integer") || kinds.contains(&"number") {
        for kw in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if let Some(v) = obj.get(kw)
                && v.as_f64().is_none()
            {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} schema {path}/{kw} must be a finite number"
                )));
            }
        }
        if let Some(v) = obj.get("multipleOf")
            && v.as_f64().filter(|m| *m > 0.0).is_none()
        {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/multipleOf must be a positive number"
            )));
        }
    }

    Ok(())
}

fn opt_u64(
    tool_name: &str,
    obj: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<u64>> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/{key} must be a non-negative integer"
            ))
        }),
    }
}

fn declared_types<'a>(
    tool_name: &str,
    obj: &'a Map<String, Value>,
    path: &str,
) -> Result<Vec<&'a str>> {
    let Some(t) = obj.get("type") else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    match t {
        Value::String(s) => out.push(s.as_str()),
        Value::Array(arr) if !arr.is_empty() => {
            for item in arr {
                let s = item.as_str().ok_or_else(|| {
                    grammar_err(format!(
                        "strict tool {tool_name:?} schema {path}/type entries must be strings"
                    ))
                })?;
                if out.contains(&s) {
                    return Err(grammar_err(format!(
                        "strict tool {tool_name:?} schema {path}/type contains duplicate {s:?}"
                    )));
                }
                out.push(s);
            }
        }
        _ => {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/type must be a string or nonempty array"
            )));
        }
    }
    for &k in &out {
        if !matches!(
            k,
            "object" | "array" | "string" | "integer" | "number" | "boolean" | "null"
        ) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema {path}/type has unknown JSON type {k:?}"
            )));
        }
    }
    Ok(out)
}

fn check_unguarded_ref_cycles(tool_name: &str, root: &Value) -> Result<()> {
    fn walk(
        tool_name: &str,
        root: &Value,
        node: &Map<String, Value>,
        id: &str,
        gray: &mut HashSet<String>,
        black: &mut HashSet<String>,
        nodes: &mut usize,
    ) -> Result<()> {
        if black.contains(id) {
            return Ok(());
        }
        *nodes += 1;
        if *nodes > MAX_SCHEMA_NODES {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema exceeds depth or node limits at {id}"
            )));
        }
        if !gray.insert(id.to_owned()) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} schema contains cyclic unguarded $ref {id:?} at {id}"
            )));
        }
        if let Some(target) = node.get("$ref").and_then(Value::as_str)
            && let Some(next) = resolve_ref_value(root, target).and_then(Value::as_object)
        {
            walk(tool_name, root, next, target, gray, black, nodes)?;
        } else if let Some(branches) = node.get("anyOf").and_then(Value::as_array) {
            for (i, branch) in branches.iter().enumerate() {
                if let Some(next) = branch.as_object() {
                    walk(
                        tool_name,
                        root,
                        next,
                        &format!("{id}/anyOf/{i}"),
                        gray,
                        black,
                        nodes,
                    )?;
                }
            }
        }
        gray.remove(id);
        black.insert(id.to_owned());
        Ok(())
    }

    let mut gray = HashSet::new();
    let mut black = HashSet::new();
    let mut nodes = 0;
    let root_obj = root.as_object().ok_or_else(|| {
        grammar_err(format!(
            "strict tool {tool_name:?} schema must be a JSON object"
        ))
    })?;
    walk(
        tool_name, root, root_obj, "#", &mut gray, &mut black, &mut nodes,
    )?;
    for defs_key in ["$defs", "definitions"] {
        let Some(defs) = root.get(defs_key).and_then(Value::as_object) else {
            continue;
        };
        for (name, def_schema) in defs {
            let Some(def_obj) = def_schema.as_object() else {
                continue;
            };
            let escaped = name.replace('~', "~0").replace('/', "~1");
            walk(
                tool_name,
                root,
                def_obj,
                &format!("#/{defs_key}/{escaped}"),
                &mut gray,
                &mut black,
                &mut nodes,
            )?;
        }
    }
    Ok(())
}

fn resolve_ref_value<'a>(root_val: &'a Value, target: &str) -> Option<&'a Value> {
    if target == "#" {
        return Some(root_val);
    }
    let pointer = target.strip_prefix('#')?;
    let name = pointer
        .strip_prefix("/$defs/")
        .or_else(|| pointer.strip_prefix("/definitions/"))?;
    if name.contains('/')
        || name
            .split('~')
            .skip(1)
            .any(|part| !part.starts_with(['0', '1']))
    {
        return None;
    }
    root_val.pointer(pointer)
}

fn unanchored_regex(pat: &str) -> String {
    // The lexer matches complete strings. Extend each top-level alternative
    // separately; a bar inside a group, class or escape is not a branch here.
    let mut out = String::from("(?:");
    let mut start = 0;
    let (mut depth, mut class, mut escape) = (0usize, false, false);
    let mut append = |branch: &str| {
        let left = branch.starts_with('^');
        let without_start = branch.strip_prefix('^').unwrap_or(branch);
        let right = without_start.strip_suffix('$').is_some_and(|prefix| {
            prefix.bytes().rev().take_while(|b| *b == b'\\').count() % 2 == 0
        });
        let body = if right {
            &without_start[..without_start.len() - 1]
        } else {
            without_start
        };
        if out.len() > 3 {
            out.push('|');
        }
        if !left {
            out.push_str("(?s:.*)");
        }
        out.push_str("(?:");
        out.push_str(body);
        out.push(')');
        if !right {
            out.push_str("(?s:.*)");
        }
    };
    for (i, c) in pat.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' => escape = true,
            '[' => class = true,
            ']' => class = false,
            '(' if !class => depth += 1,
            ')' if !class => depth = depth.saturating_sub(1),
            '|' if !class && depth == 0 => {
                append(&pat[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    append(&pat[start..]);
    out.push(')');
    out
}

fn value_matches_kind(v: &Value, kind: &str) -> bool {
    match kind {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "integer" => v.as_number().is_some_and(|n| {
            n.is_i64()
                || n.is_u64()
                || n.as_f64()
                    .is_some_and(|x| x.is_finite() && x.fract() == 0.0)
        }),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => false,
    }
}

fn collect_rebased_defs(root_schema: &Value) -> Map<String, Value> {
    let mut root_obj = root_schema.as_object().unwrap().clone();
    let mut defs = Map::new();
    for key in ["$defs", "definitions"] {
        if let Some(d) = root_obj.remove(key).and_then(|v| v.as_object().cloned()) {
            for (k, v) in d {
                defs.insert(format!("{key}:{k}"), rewrite_refs(&v));
            }
        }
    }
    let clean_root = rewrite_refs(&Value::Object(root_obj));
    defs.insert("__strict_root__".into(), clean_root);
    defs
}

fn contains_reference(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.contains_key("$ref") || m.values().any(contains_reference),
        Value::Array(a) => a.iter().any(contains_reference),
        _ => false,
    }
}

/// Attach rebased `$defs` (including `__strict_root__` for `$ref: "#"`) to an isolated
/// parameter JSON subschema so `TopLevelGrammar::from_json_schema` resolves all local `$ref`s.
fn embed_subschema_with_defs(root_schema: &Value, subschema: &Value) -> Value {
    if !contains_reference(subschema) {
        return subschema.clone();
    }
    let defs = collect_rebased_defs(root_schema);
    let mut out = rewrite_refs(subschema);
    if let Some(obj) = out.as_object_mut() {
        obj.remove("$defs");
        obj.remove("definitions");
        if !defs.is_empty() {
            obj.insert("$defs".into(), Value::Object(defs));
        }
    }
    out
}

/// Rebase `#` references onto `#/$defs/__strict_root__` so a strict schema can be
/// embedded inside TMLv0's `{"name": ..., "args": ...}` wrapper without breaking `$ref: "#"`.
fn embed_schema_in_tml_wrapper(tool_name: &str, schema: &Value) -> Value {
    let defs = collect_rebased_defs(schema);
    let clean_root = defs.get("__strict_root__").cloned().unwrap();
    let mut wrapper = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "const": tool_name },
            "args": clean_root,
        },
        "required": ["name", "args"],
        "additionalProperties": false,
    });
    if !defs.is_empty() {
        wrapper
            .as_object_mut()
            .unwrap()
            .insert("$defs".into(), Value::Object(defs));
    }
    wrapper
}

fn rewrite_refs(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                if k == "$ref"
                    && let Some(s) = v.as_str()
                {
                    let mapped = if s == "#" {
                        "#/$defs/__strict_root__".to_owned()
                    } else if let Some(rest) = s.strip_prefix("#/$defs/") {
                        format!("#/$defs/$defs:{rest}")
                    } else if let Some(rest) = s.strip_prefix("#/definitions/") {
                        format!("#/$defs/definitions:{rest}")
                    } else {
                        s.to_owned()
                    };
                    out.insert(k.clone(), Value::String(mapped));
                } else {
                    out.insert(k.clone(), rewrite_refs(v));
                }
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(rewrite_refs).collect()),
        other => other.clone(),
    }
}

#[derive(Clone)]
enum SplitBranches {
    Pure {
        string_branch: Option<Value>,
        nonstring_branch: Option<Value>,
    },
    UnsupportedMixed,
}

struct ExpansionBudget(usize);

impl std::io::Write for ExpansionBudget {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_sub(data.len())
            .ok_or_else(|| std::io::Error::other("strict schema expansion limit"))?;
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl ExpansionBudget {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.0 = self
            .0
            .checked_sub(bytes)
            .ok_or_else(|| grammar_err("strict schema expansion limit"))?;
        Ok(())
    }
    fn value(&mut self, v: &Value) -> Result<()> {
        serde_json::to_writer(self, v).map_err(|_| grammar_err("strict schema expansion limit"))
    }
    // Walk exactly the reference/anyOf edges the wire splitter unfolds. An
    // alias used twice costs twice; no intermediate schemas are materialized.
    fn split(&mut self, root: &Value, v: &Value, depth: usize) -> Result<()> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(grammar_err("strict schema expansion depth limit"));
        }
        self.charge(std::mem::size_of::<Value>())?;
        if let Some(reference) = v.get("$ref").and_then(Value::as_str) {
            return self.split(
                root,
                resolve_ref_value(root, reference)
                    .ok_or_else(|| grammar_err("unresolved strict property reference"))?,
                depth + 1,
            );
        }
        if let Some(branches) = v.get("anyOf").and_then(Value::as_array) {
            for branch in branches {
                self.split(root, branch, depth + 1)?;
            }
            return Ok(());
        }
        self.value(v)?;
        if v.get("type").is_some_and(Value::is_array) {
            self.value(v)?;
        }
        Ok(())
    }
}

fn wire_hints<'a>(
    presentation: Option<&'a mut crate::output::tagged::OutputConfig>,
    name: &str,
    has_fields: bool,
) -> Result<Option<&'a mut Map<String, Value>>> {
    let Some(output) = presentation.filter(|_| has_fields) else {
        return Ok(None);
    };
    output
        .tools
        .iter_mut()
        .find(|tool| tool["function"]["name"].as_str() == Some(name))
        .and_then(|tool| tool.pointer_mut("/function/parameters/properties"))
        .and_then(Value::as_object_mut)
        .map(Some)
        .ok_or_else(|| grammar_err("strict tool has no matching output descriptor"))
}

// The tag parser needs only the wire representation chosen by the grammar.
// Its private output descriptor is distinct from the prompt and final schema.
fn set_wire_hint(
    hints: &mut Option<&mut Map<String, Value>>,
    key: &str,
    schema: &Value,
    raw_string: bool,
) {
    if schema["type"].is_string() {
        return;
    }
    if let Some(property) = hints.as_deref_mut().and_then(|props| props.get_mut(key)) {
        *property = if raw_string {
            serde_json::json!({"type":"string"})
        } else {
            serde_json::json!({"type":["object","array","number","boolean","null"]})
        };
    }
}

fn split_top_property(root: &Value, prop_schema: &Value) -> SplitBranches {
    let mut resolved = prop_schema;
    let mut remaining = MAX_SCHEMA_DEPTH;
    while let Some(r) = resolved.get("$ref").and_then(Value::as_str) {
        if remaining == 0 {
            return SplitBranches::UnsupportedMixed;
        }
        remaining -= 1;
        let Some(v) = resolve_ref_value(root, r) else {
            return SplitBranches::UnsupportedMixed;
        };
        resolved = v;
    }
    let obj = resolved.as_object().unwrap();
    if let Some(any_of) = obj.get("anyOf").and_then(Value::as_array) {
        let mut s_alts = Vec::new();
        let mut ns_alts = Vec::new();
        for b in any_of {
            match split_top_property(root, b) {
                SplitBranches::Pure {
                    string_branch,
                    nonstring_branch,
                } => {
                    if let Some(sb) = string_branch {
                        s_alts.push(sb);
                    }
                    if let Some(nb) = nonstring_branch {
                        ns_alts.push(nb);
                    }
                }
                SplitBranches::UnsupportedMixed => return SplitBranches::UnsupportedMixed,
            }
        }
        let s_out = match s_alts.len() {
            0 => None,
            1 => Some(s_alts.remove(0)),
            _ => Some(serde_json::json!({"anyOf": s_alts})),
        };
        let ns_out = match ns_alts.len() {
            0 => None,
            1 => Some(ns_alts.remove(0)),
            _ => Some(serde_json::json!({"anyOf": ns_alts})),
        };
        return SplitBranches::Pure {
            string_branch: s_out,
            nonstring_branch: ns_out,
        };
    }
    if let Some(c) = obj.get("const") {
        return if c.is_string() {
            SplitBranches::Pure {
                string_branch: Some(resolved.clone()),
                nonstring_branch: None,
            }
        } else {
            SplitBranches::Pure {
                string_branch: None,
                nonstring_branch: Some(resolved.clone()),
            }
        };
    }
    if let Some(en) = obj.get("enum").and_then(Value::as_array) {
        let strs: Vec<_> = en.iter().filter(|v| v.is_string()).cloned().collect();
        let nonstrs: Vec<_> = en.iter().filter(|v| !v.is_string()).cloned().collect();
        return SplitBranches::Pure {
            string_branch: (!strs.is_empty()).then(|| serde_json::json!({"enum": strs})),
            nonstring_branch: (!nonstrs.is_empty()).then(|| serde_json::json!({"enum": nonstrs})),
        };
    }
    match obj.get("type") {
        Some(Value::String(s)) if s == "string" => SplitBranches::Pure {
            string_branch: Some(resolved.clone()),
            nonstring_branch: None,
        },
        Some(Value::String(_)) => SplitBranches::Pure {
            string_branch: None,
            nonstring_branch: Some(resolved.clone()),
        },
        Some(Value::Array(arr)) => {
            let has_str = arr.iter().any(|v| v.as_str() == Some("string"));
            let others: Vec<_> = arr
                .iter()
                .filter(|v| v.as_str() != Some("string"))
                .cloned()
                .collect();
            let s_branch = has_str.then(|| {
                let mut m = obj.clone();
                m.insert("type".into(), Value::String("string".into()));
                Value::Object(m)
            });
            let ns_branch = (!others.is_empty()).then(|| {
                let mut m = obj.clone();
                for k in ["minLength", "maxLength", "pattern"] {
                    m.remove(k);
                }
                m.insert(
                    "type".into(),
                    if others.len() == 1 {
                        others[0].clone()
                    } else {
                        Value::Array(others)
                    },
                );
                Value::Object(m)
            });
            SplitBranches::Pure {
                string_branch: s_branch,
                nonstring_branch: ns_branch,
            }
        }
        _ => SplitBranches::UnsupportedMixed,
    }
}

/// Build an isolated subgrammar matching a raw string parameter body AND consuming its
/// closing wire delimiter via lazy `[suffix=...]` so `llguidance` needs zero token backtracking.
#[derive(Clone, Copy)]
enum ParameterClose<'a> {
    Text(&'a str),
    Dsml(u32, &'a str),
}

fn qwen_parameter_boundary(text: &str) -> bool {
    text.match_indices('<').any(|(at, _)| {
        let tail = text[at + 1..].trim_start();
        if let Some(tail) = tail.strip_prefix('/') {
            return tail
                .trim_start()
                .strip_prefix("parameter")
                .is_some_and(|tail| tail.trim_start().starts_with('>'));
        }
        tail.strip_prefix("parameter")
            .is_some_and(|tail| tail.trim_start().starts_with('='))
    })
}

fn raw_string_has_forbidden_boundary(
    text: &str,
    close_delimiter: ParameterClose<'_>,
    forbidden_substrings: &[&str],
) -> bool {
    forbidden_substrings.iter().any(|bad| text.contains(bad))
        || matches!(
            close_delimiter,
            ParameterClose::Text(close)
                if close.ends_with("</parameter>") && qwen_parameter_boundary(text)
        )
}

fn build_raw_string_with_close_subgrammar(
    tool_name: &str,
    sub_name: String,
    s_schema: &Value,
    close_delimiter: ParameterClose<'_>,
    forbidden_substrings: &[&str],
) -> Result<GrammarWithLexer> {
    let mut rules = String::new();
    let mut counter = 0usize;
    let start_expr = lower_raw_string_with_close_expr(
        tool_name,
        s_schema,
        close_delimiter,
        forbidden_substrings,
        &mut rules,
        &mut counter,
    )?;
    rules.push_str(&format!("start: {start_expr}\n"));
    let mut g = GrammarWithLexer::from_lark(rules);
    g.name = Some(sub_name);
    Ok(g)
}

fn lower_raw_string_with_close_expr(
    tool_name: &str,
    s_schema: &Value,
    close_delimiter: ParameterClose<'_>,
    forbidden_substrings: &[&str],
    rules: &mut String,
    counter: &mut usize,
) -> Result<String> {
    let id = *counter;
    *counter += 1;
    let rname = format!("rstr_{id}");
    let close_lit = match close_delimiter {
        ParameterClose::Text(s) => serde_json::to_string(s).unwrap(),
        ParameterClose::Dsml(id, tag) => {
            format!("\"</\" <[{id}]> {}", serde_json::to_string(tag).unwrap())
        }
    };
    let obj = s_schema.as_object().unwrap();

    if let Some(any_of) = obj.get("anyOf").and_then(Value::as_array) {
        let mut alts = Vec::new();
        for b in any_of {
            alts.push(lower_raw_string_with_close_expr(
                tool_name,
                b,
                close_delimiter,
                forbidden_substrings,
                rules,
                counter,
            )?);
        }
        rules.push_str(&format!("{rname}: {}\n", alts.join(" | ")));
        return Ok(rname);
    }
    if let Some(s) = obj.get("const").and_then(Value::as_str) {
        if raw_string_has_forbidden_boundary(s, close_delimiter, forbidden_substrings) {
            return Err(grammar_err(format!(
                "strict tool {tool_name:?} string const contains wire delimiter text"
            )));
        }
        rules.push_str(&format!(
            "{rname}: {} {close_lit}\n",
            serde_json::to_string(s).unwrap()
        ));
        return Ok(rname);
    }
    if let Some(en) = obj.get("enum").and_then(Value::as_array) {
        let mut alts = Vec::with_capacity(en.len());
        for v in en {
            let s = v.as_str().unwrap();
            if raw_string_has_forbidden_boundary(s, close_delimiter, forbidden_substrings) {
                return Err(grammar_err(format!(
                    "strict tool {tool_name:?} string enum contains wire delimiter text"
                )));
            }
            alts.push(serde_json::to_string(s).unwrap());
        }
        rules.push_str(&format!("{rname}: ({}) {close_lit}\n", alts.join(" | ")));
        return Ok(rname);
    }

    let min_len = obj.get("minLength").and_then(Value::as_u64);
    let max_len = obj.get("maxLength").and_then(Value::as_u64);
    let pat = obj.get("pattern").and_then(Value::as_str);

    let mut clauses = Vec::new();
    if let ParameterClose::Text(close) = close_delimiter {
        let mut forbid_parts = forbidden_substrings
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>();
        if close.ends_with("</parameter>") {
            forbid_parts.extend([
                "/<\\s*\\/\\s*parameter\\s*>/".to_owned(),
                "/<\\s*parameter\\s*=/".to_owned(),
            ]);
        }
        let forbid_expr = forbid_parts.join(" | ");
        clauses.push(format!(
            "(/(?s:.*)/ & ~(/(?s:.*)/ ({forbid_expr}) /(?s:.*)/))"
        ));
    }
    if min_len.is_some() || max_len.is_some() {
        let lo = min_len.unwrap_or(0);
        let q = match max_len {
            Some(hi) => format!("{{{lo},{hi}}}"),
            None => format!("{{{lo},}}"),
        };
        clauses.push(format!("/(?s:.){q}/"));
    }
    if let Some(p) = pat {
        let unanch = llguidance::regex_to_lark(&unanchored_regex(p), "");
        clauses.push(format!("/{unanch}/"));
    }

    let term = format!("RAW_BODY_{id}");
    // A DSML close contains the dedicated DSML token, which text lexemes cannot
    // consume. Avoid intersecting every raw value with a redundant decoded-marker
    // exclusion; on a checkpoint vocabulary that product can exhaust the lexer
    // expression bound while computing a generation mask.
    let body = if clauses.is_empty() {
        "/(?s:.*)/".to_owned()
    } else {
        clauses.join(" & ")
    };
    match close_delimiter {
        ParameterClose::Text(_) => rules.push_str(&format!(
            "{term}: ({body})\n{rname}[suffix={close_lit}]: {term}\n"
        )),
        ParameterClose::Dsml(id, tag) => rules.push_str(&format!(
            "{term}: ({body}) \"</\"\n{rname}: {term} <[{id}]> {}\n",
            serde_json::to_string(tag).unwrap()
        )),
    }
    Ok(rname)
}

fn push_json_param_subgrammar(
    subgrammars: &mut Vec<GrammarWithLexer>,
    sub_name: String,
    root_schema: &Value,
    param_schema: &Value,
) -> String {
    let embedded = embed_subschema_with_defs(root_schema, param_schema);
    let mut g = TopLevelGrammar::from_json_schema(embedded)
        .grammars
        .remove(0);
    g.name = Some(sub_name.clone());
    subgrammars.push(g);
    format!("@{sub_name}")
}

fn push_plain_answer_subgrammar(
    compiler: &GrammarCompiler,
    subgrammars: &mut Vec<GrammarWithLexer>,
    forbidden_literals: &[&str],
    excluded_token_ids: &[u32],
) {
    let forbid = forbidden_literals
        .iter()
        .map(|s| serde_json::to_string(s).unwrap())
        .collect::<Vec<_>>()
        .join(" | ");
    let specials: Vec<_> = compiler
        .special_tokens
        .iter()
        .filter(|(id, spelling)| {
            !excluded_token_ids.contains(id)
                && !compiler.eos_ids.contains(id)
                && !forbidden_literals
                    .iter()
                    .any(|lit| spelling.contains(lit) || spelling.chars().any(|c| lit.contains(c)))
        })
        .map(|(id, _)| id.to_string())
        .collect();
    let mut lark = format!("PLAIN_SEG: (/(?s:.+)/ & ~(/(?s:.*)/ ({forbid}) /(?s:.*)/))\n");
    if specials.is_empty() {
        lark.push_str("start: PLAIN_SEG?\n");
    } else {
        lark.push_str(&format!(
            "start: PLAIN_SEG? (<[{}]> PLAIN_SEG?)*\n",
            specials.join(",")
        ));
    }
    let mut g = GrammarWithLexer::from_lark(lark);
    g.name = Some("plain_answer".into());
    subgrammars.push(g);
}

// The tagged parsers require a name followed by a parameter or closing tag.
// The opener alone can be ordinary text, including in quoted examples.
fn push_tagged_answer(subgrammars: &mut Vec<GrammarWithLexer>, call_prefix: &str) {
    let mut g = GrammarWithLexer::from_lark(format!(
        "PLAIN: (/(?s:.+)/ & ~(/(?s:.*)/ /{call_prefix}/ /(?s:.*)/))\nstart: PLAIN?\n"
    ));
    g.name = Some("plain_answer".into());
    subgrammars.push(g);
}

impl GrammarCompiler {
    #[cfg(test)]
    pub(crate) fn compile_with_strict_tools(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: Option<&OutputEnvelope>,
        plan: &ToolGrammarPlan,
    ) -> Result<RequestGrammar> {
        for tool in &plan.tools {
            if let Some(schema) = tool.strict_schema {
                validate_strict_tool_schema(self, &tool.name, schema)?;
            }
        }
        self.compile_presented_strict_tools(answer_constraint, envelope, plan, None)
    }

    pub(crate) fn compile_presented_strict_tools(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: Option<&OutputEnvelope>,
        plan: &ToolGrammarPlan,
        mut presentation: Option<&mut crate::output::tagged::OutputConfig>,
    ) -> Result<RequestGrammar> {
        let Some((kind, wrapping_newlines, ref sep)) = plan.syntax else {
            return Err(grammar_err(
                "strict function tool schemas require a prepared model tool parser",
            ));
        };
        let Some(envelope) = envelope else {
            return Err(grammar_err(
                "strict function tool schemas require a parsed output envelope",
            ));
        };
        // Request admission has already validated every strict schema and compiled
        // its completed-argument validator. This pass only budgets and composes the
        // parser-specific wire grammar.
        let mut budget = ExpansionBudget(MAX_GRAMMAR_BYTES);
        for tool in &plan.tools {
            let Some(schema) = tool.strict_schema else {
                continue;
            };
            budget.value(schema)?;
            if kind == ToolParserKind::TmlJson {
                // The wrapper can only rename references, not unfold them.
                let mut structural = ExpansionBudget(MAX_GRAMMAR_BYTES);
                for field in [
                    "$defs",
                    "definitions",
                    "type",
                    "properties",
                    "required",
                    "items",
                    "prefixItems",
                    "anyOf",
                ] {
                    if let Some(v) = schema.get(field) {
                        structural.split(schema, v, 0)?;
                    }
                }
            }
        }
        for tool in &plan.tools {
            let Some(schema) = tool.strict_schema else {
                continue;
            };
            if let Some(output) = presentation.as_deref_mut() {
                let function = output
                    .tools
                    .iter_mut()
                    .filter_map(|t| t.get_mut("function"))
                    .find(|f| f["name"].as_str() == Some(&tool.name))
                    .ok_or_else(|| grammar_err("missing strict output tool"))?;
                function["strict"] = Value::Bool(true);
            }
            let before = budget.0;
            budget.value(schema)?;
            let source_bytes = before - budget.0;
            if kind == ToolParserKind::TmlJson {
                budget.charge(source_bytes)?;
            } else if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                for prop in props.values() {
                    if contains_reference(prop) {
                        budget.charge(source_bytes)?;
                    }
                    budget.split(schema, prop, 0)?;
                }
            }
        }
        match kind {
            ToolParserKind::TmlJson => {
                self.compile_strict_tml(answer_constraint, envelope, plan, &mut budget)
            }
            ToolParserKind::Dsml => {
                self.compile_strict_dsml(answer_constraint, envelope, plan, sep == " ")
            }
            ToolParserKind::QwenCoder => self.compile_strict_qwen(
                answer_constraint,
                envelope,
                plan,
                wrapping_newlines > 0,
                presentation,
                &mut budget,
            ),
            ToolParserKind::GlmArgumentTags => {
                self.compile_strict_glm(answer_constraint, envelope, plan, presentation)
            }
            other => Err(grammar_err(format!(
                "strict function tool schemas are not supported for {other:?} tool syntax"
            ))),
        }
    }

    fn prepare_answer_subgrammar(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        forbid_lark: bool,
        subgrammars: &mut Vec<GrammarWithLexer>,
    ) -> Result<bool> {
        match answer_constraint {
            None => Ok(false),
            Some(AnswerConstraint::Lark(_)) if forbid_lark => Err(grammar_err(
                "custom Lark response grammars cannot be combined with strict function tools on tag-delimited models",
            )),
            Some(AnswerConstraint::Lark(source)) => {
                validate_lark(source)?;
                let mut g = GrammarWithLexer::from_lark(source.to_owned());
                g.name = Some("answer".into());
                subgrammars.push(g);
                Ok(true)
            }
            Some(AnswerConstraint::JsonObject) => {
                let mut g =
                    TopLevelGrammar::from_json_schema(serde_json::json!({"type": "object"}))
                        .grammars
                        .remove(0);
                g.name = Some("answer".into());
                subgrammars.push(g);
                Ok(true)
            }
            Some(AnswerConstraint::Json(schema)) => {
                if schema.to_string().len() > MAX_GRAMMAR_BYTES {
                    return Err(grammar_err(format!(
                        "JSON schema exceeds {MAX_GRAMMAR_BYTES} bytes"
                    )));
                }
                let mut nodes = 0;
                validate_schema_nodes(schema, "$", 0, &mut nodes)?;
                let mut g = TopLevelGrammar::from_json_schema(schema.clone())
                    .grammars
                    .remove(0);
                g.name = Some("answer".into());
                subgrammars.push(g);
                Ok(true)
            }
        }
    }

    fn compile_strict_tml(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        budget: &mut ExpansionBudget,
    ) -> Result<RequestGrammar> {
        use crate::output::tagged::OutputStart;
        let recipient = envelope
            .recipient
            .as_ref()
            .ok_or_else(|| grammar_err("missing TMLv0 recipient envelope"))?;
        let restart = self.marker("<|message_model|>");
        let end = self.marker("<|end_message|>");
        let mut rules = self.region("thinking_body", "<|end_message|>");
        let mut subgrammars = Vec::new();
        let has_structured_answer =
            self.prepare_answer_subgrammar(answer_constraint, false, &mut subgrammars)?;
        if has_structured_answer {
            rules.push_str(&format!(
                "thinking: {} thinking_body {restart}\nanswer: {} @answer {end}\n",
                self.marker("<|content_thinking|>"),
                self.marker("<|content_text|>")
            ));
        } else {
            rules.push_str(&self.region("plain_text_body", "<|end_message|>"));
            rules.push_str(&format!(
                "thinking: {} thinking_body {restart}\nanswer: {} plain_text_body\n",
                self.marker("<|content_thinking|>"),
                self.marker("<|content_text|>")
            ));
        }
        let mut tool_rules = Vec::new();
        for (index, tool) in plan.tools.iter().enumerate() {
            let wrapper = match &tool.strict_schema {
                Some(schema) => embed_schema_in_tml_wrapper(&tool.name, schema),
                None => serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": { "const": tool.name },
                        "args": { "type": "object" }
                    },
                    "required": ["name", "args"],
                    "additionalProperties": false,
                }),
            };
            budget.value(&wrapper)?;
            let mut payload = TopLevelGrammar::from_json_schema(wrapper)
                .grammars
                .remove(0);
            payload.name = Some(format!("payload_{index}"));
            subgrammars.push(payload);
            rules.push_str(&format!(
                "tool_{index}: {} {} @payload_{index} {end}\n",
                self.marker(&tool.name),
                self.marker("<|content_invoke_tool_json|>")
            ));
            tool_rules.push(format!("tool_{index}"));
        }
        if tool_rules.is_empty() {
            return Err(grammar_err(
                "strict tool grammar requires at least one active tool",
            ));
        }
        rules.push_str(&format!("tool: {}\n", tool_rules.join(" | ")));
        if plan.single_call {
            rules.push_str(&format!("tools: tool ({restart} thinking* answer)?\n"));
        } else {
            rules.push_str(&format!(
                "tools: tool ({restart} thinking* tool)* ({restart} thinking* answer)?\n"
            ));
        }
        let ending = if plan.require_call {
            "tools"
        } else {
            "(answer | tools)"
        };
        rules.push_str(&format!("tail: thinking* {ending}\n"));
        match recipient.start {
            OutputStart::RecipientHeader => rules.push_str("start: tail\n"),
            OutputStart::RecipientFrame => rules.push_str(&format!("start: {restart} tail\n")),
            OutputStart::Reasoning => {
                rules.push_str(&format!("start: thinking_body {restart} tail\n"))
            }
            OutputStart::Content => {
                if plan.require_call {
                    return Err(grammar_err(
                        "required tool choice cannot continue a plain TMLv0 text frame",
                    ));
                }
                if has_structured_answer {
                    rules.push_str(&format!("start: @answer {end}\n"));
                } else {
                    rules.push_str("start: plain_text_body\n");
                }
            }
            OutputStart::ToolBody => {
                return Err(grammar_err(
                    "strict tool grammar cannot continue an already-opened TMLv0 tool body",
                ));
            }
        }
        self.finish_composed(rules, subgrammars)
    }

    fn compile_strict_dsml(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        spaced: bool,
    ) -> Result<RequestGrammar> {
        let (dsml_id, dsml_str) = plan.dsml_token.as_ref().ok_or_else(|| {
            grammar_err("DeepSeek DSML strict tool grammar requires the DSML token")
        })?;
        let (block_tag, invoke_tag, param_tag) = if spaced {
            (" calls", " invoke", " parameter")
        } else {
            ("tool_calls", "invoke", "parameter")
        };
        let mut rules = String::from("ws: /[ \\t\\r\\n]*/\n");
        let mut subgrammars = Vec::new();
        let has_structured_answer =
            self.prepare_answer_subgrammar(answer_constraint, true, &mut subgrammars)?;
        if !has_structured_answer && !plan.require_call {
            push_plain_answer_subgrammar(self, &mut subgrammars, &[dsml_str.as_str()], &[*dsml_id]);
        }

        let close_block = format!(
            "\"</\" <[{dsml_id}]> {}",
            serde_json::to_string(&format!("{block_tag}>")).unwrap()
        );
        let close_invoke = format!(
            "\"</\" <[{dsml_id}]> {}",
            serde_json::to_string(&format!("{invoke_tag}>")).unwrap()
        );
        let close_param_tok = format!(
            "\"</\" <[{dsml_id}]> {}",
            serde_json::to_string(&format!("{param_tag}>")).unwrap()
        );
        let param_tail = format!("{param_tag}>");

        let mut invoke_alts = Vec::new();
        let mut strict_names = Vec::new();

        for (t_idx, tool) in plan.tools.iter().enumerate() {
            if let Some(schema) = &tool.strict_schema {
                strict_names.push(tool.name.clone());
                let inv_open = format!(
                    "\"<\" <[{dsml_id}]> {}",
                    serde_json::to_string(&format!("{invoke_tag} name=\"{}\">", tool.name))
                        .unwrap()
                );
                let empty = Map::new();
                let props = schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty);
                let mut param_seq = Vec::with_capacity(props.len());
                for (p_idx, (prop_name, prop_schema)) in props.iter().enumerate() {
                    let p_head = format!(
                        "\"<\" <[{dsml_id}]> {}",
                        serde_json::to_string(&format!("{param_tag} name=\"{prop_name}\" string="))
                            .unwrap()
                    );
                    let val_expr = match split_top_property(schema, prop_schema) {
                        SplitBranches::Pure {
                            string_branch,
                            nonstring_branch,
                        } => {
                            let mut parts = Vec::new();
                            if let Some(sb) = string_branch {
                                let s_sub = format!("dsml_s_{t_idx}_{p_idx}");
                                let g = build_raw_string_with_close_subgrammar(
                                    &tool.name,
                                    s_sub.clone(),
                                    &sb,
                                    ParameterClose::Dsml(*dsml_id, &param_tail),
                                    &[dsml_str.as_str()],
                                )?;
                                subgrammars.push(g);
                                parts.push(format!("\"\\\"true\\\">\" @{s_sub}"));
                            }
                            if let Some(nb) = nonstring_branch {
                                let j_ref = push_json_param_subgrammar(
                                    &mut subgrammars,
                                    format!("dsml_j_{t_idx}_{p_idx}"),
                                    schema,
                                    &nb,
                                );
                                parts.push(format!("\"\\\"false\\\">\" {j_ref} {close_param_tok}"));
                            }
                            parts.join(" | ")
                        }
                        SplitBranches::UnsupportedMixed => {
                            return Err(grammar_err(format!(
                                "strict tool {:?} property {prop_name:?} cannot be lowered to DSML parameter syntax",
                                tool.name
                            )));
                        }
                    };
                    let pname = format!("dsml_param_{t_idx}_{p_idx}");
                    rules.push_str(&format!("{pname}: {p_head} ({val_expr})\n"));
                    param_seq.push(pname);
                }
                let iname = format!("dsml_invoke_{t_idx}");
                if param_seq.is_empty() {
                    rules.push_str(&format!("{iname}: {inv_open} ws {close_invoke}\n"));
                } else {
                    rules.push_str(&format!(
                        "{iname}: {inv_open} ws {} ws {close_invoke}\n",
                        param_seq.join(" ws ")
                    ));
                }
                invoke_alts.push(iname);
            } else {
                let inv_open = format!(
                    "\"<\" <[{dsml_id}]> {}",
                    serde_json::to_string(&format!("{invoke_tag} name=\"{}\">", tool.name))
                        .unwrap()
                );
                invoke_alts.push(format!("({inv_open} @dsml_free_params {close_invoke})"));
            }
        }

        let need_free_params =
            plan.tools.iter().any(|t| t.strict_schema.is_none()) || plan.allow_undeclared;
        if need_free_params {
            let p_open = format!(
                "\"<\" <[{dsml_id}]> {}",
                serde_json::to_string(&format!("{param_tag} name=\"")).unwrap()
            );
            let param_end = serde_json::to_string(&param_tail).unwrap();
            let forbid = serde_json::to_string(dsml_str).unwrap();
            let free_lark = format!(
                "ws: /[ \\t\\r\\n]*/\n\
                 DSML_FREE_KEY: /[^\"\\r\\n]+/\n\
                 DSML_FREE_BODY: (/(?s:.*)/ & ~(/(?s:.*)/ {forbid} /(?s:.*)/)) \"</\"\n\
                 dsml_free_val: DSML_FREE_BODY <[{dsml_id}]> {param_end}\n\
                 dsml_free_param: {p_open} DSML_FREE_KEY \"\\\" string=\\\"\" (\"true\" | \"false\") \"\\\">\" dsml_free_val\n\
                 start: (ws dsml_free_param)* ws\n"
            );
            let mut fg = GrammarWithLexer::from_lark(free_lark);
            fg.name = Some("dsml_free_params".into());
            subgrammars.push(fg);
        }
        if plan.allow_undeclared {
            let excl = strict_names
                .iter()
                .map(|n| serde_json::to_string(n).unwrap())
                .collect::<Vec<_>>()
                .join(" | ");
            let other_name_lark = format!("OTHER: (/[^\"<>\\s]+/ & ~({excl}))\nstart: OTHER\n");
            let mut og = GrammarWithLexer::from_lark(other_name_lark);
            og.name = Some("dsml_other_name".into());
            subgrammars.push(og);
            let other_open = format!(
                "\"<\" <[{dsml_id}]> {}",
                serde_json::to_string(&format!("{invoke_tag} name=\"")).unwrap()
            );
            invoke_alts.push(format!(
                "({other_open} @dsml_other_name \"\\\">\" @dsml_free_params {close_invoke})"
            ));
        }

        rules.push_str(&format!("dsml_invoke: {}\n", invoke_alts.join(" | ")));
        if plan.single_call {
            rules.push_str(&format!(
                "tool_after_open: ws dsml_invoke ws {close_block}\n"
            ));
        } else {
            rules.push_str(&format!(
                "tool_after_open: ws dsml_invoke (ws dsml_invoke)* ws {close_block}\n"
            ));
        }

        // The prose token ends at `<`; only the declared DSML token can open a call.
        let open_block_tok = format!(
            "\"<\" <[{dsml_id}]> {}",
            serde_json::to_string(&format!("{block_tag}>")).unwrap()
        );
        if has_structured_answer {
            rules.push_str(&format!("tools_entry: {open_block_tok} tool_after_open\n"));
        } else {
            let forbid = serde_json::to_string(dsml_str).unwrap();
            let open_tail = serde_json::to_string(&format!("{block_tag}>")).unwrap();
            let pre_lark = format!(
                "PRE_BODY: (/(?s:.*)/ & ~(/(?s:.*)/ {forbid} /(?s:.*)/)) \"<\"\n\
                 start: PRE_BODY <[{dsml_id}]> {open_tail}\n"
            );
            let mut pg = GrammarWithLexer::from_lark(pre_lark);
            pg.name = Some("dsml_pre_tool".into());
            subgrammars.push(pg);
            rules.push_str("tools_entry: @dsml_pre_tool tool_after_open\n");
        }

        let body_branches = if plan.require_call {
            "tools_entry".to_owned()
        } else if has_structured_answer {
            "(@answer | tools_entry)".to_owned()
        } else {
            "(@plain_answer | tools_entry)".to_owned()
        };

        if envelope.tool_body_start {
            rules.push_str("start: tool_after_open ws\n");
        } else if let Some(end) = envelope.reasoning_end {
            let mut r_lark = self.region("reasoning", end);
            r_lark.push_str("start: reasoning\n");
            let mut rg = GrammarWithLexer::from_lark(r_lark);
            rg.name = Some("dsml_reasoning".into());
            subgrammars.push(rg);
            rules.push_str(&format!("start: @dsml_reasoning ws {body_branches} ws\n"));
        } else {
            rules.push_str(&format!("start: ws {body_branches} ws\n"));
        }
        self.finish_composed(rules, subgrammars)
    }

    fn non_marker_lt_regex(tags: &[&str], extra_first: &str) -> String {
        fn step(tails: &[&str], extra: &str) -> String {
            let mut first_chars: Vec<char> = Vec::new();
            for t in tails {
                let c = t.chars().next().unwrap();
                if !first_chars.contains(&c) {
                    first_chars.push(c);
                }
            }
            let mut excl = String::from("<");
            excl.push_str(extra);
            for &c in &first_chars {
                if c == '/' {
                    excl.push_str(r"\/");
                } else {
                    excl.push(c);
                }
            }
            let mut alts = vec![format!("[^{excl}]")];
            for c in first_chars {
                let esc = if c == '/' {
                    r"\/".to_owned()
                } else {
                    c.to_string()
                };
                let sub: Vec<&str> = tails.iter().filter_map(|t| t.strip_prefix(c)).collect();
                if sub.iter().any(|s| s.is_empty()) {
                    continue;
                }
                alts.push(format!("{esc}{}", step(&sub, "")));
            }
            format!("(?:{})", alts.join("|"))
        }
        format!("<{}", step(tags, extra_first))
    }

    fn proper_lt_prefixes(tags: &[&str]) -> String {
        let mut out = Vec::new();
        for tag in tags {
            for end in 1..tag.len() {
                let p = tag[..end].replace('/', r"\/");
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        format!("(?:{})", out.join("|"))
    }

    fn compile_strict_qwen(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        wrapping_newlines: bool,
        mut presentation: Option<&mut crate::output::tagged::OutputConfig>,
        budget: &mut ExpansionBudget,
    ) -> Result<RequestGrammar> {
        let mut rules = String::from("ws: /[ \\t\\r\\n]*/\n");
        let q_pre = Self::proper_lt_prefixes(&["<tool_call>", "<function=", "</tool_call>"]);
        let q_non_marker_lt =
            Self::non_marker_lt_regex(&["tool_call>", "function=", "/tool_call>"], r"\s");
        let q_fn_direct_lt = Self::non_marker_lt_regex(
            &["tool_call>", "function=", "/tool_call>", "/function>"],
            r"p\s",
        );
        let q_tc_pre = q_pre.clone();
        let q_fn_pre = Self::proper_lt_prefixes(&[
            "<parameter=",
            "</function>",
            "<tool_call>",
            "<function=",
            "</tool_call>",
        ]);
        rules.push_str(&format!(
            "QWEN_MENTION_HEADER: /[^<>]*>/\n\
             QWEN_MENTION_SPACE: /[^<>]*>[ \\t\\r\\n]+/\n\
             QWEN_MENTION_TEXT: /[^<>]*>[ \\t\\r\\n]*[^<\\s]/\n\
             QWEN_NONCALL_FUNCTION_LT: /(?:{q_fn_direct_lt}|{q_fn_pre}{q_pre}*{q_non_marker_lt})/\n\
             QWEN_FN_TO_TC: /(?:<tool_call>|{q_fn_pre}{q_pre}*<tool_call>)/\n\
             QWEN_FN_TO_FN: /(?:<function=|{q_fn_pre}{q_pre}*<function=)/\n\
             QWEN_FN_TO_CLOSE: /(?:<\\/tool_call>|{q_fn_pre}{q_pre}*<\\/tool_call>)/\n\
             QWEN_NONCALL_TOOL_LT: /(?:{q_non_marker_lt}|{q_tc_pre}{q_pre}*{q_non_marker_lt})/\n\
             QWEN_TC_TO_TC: /(?:<tool_call>|{q_tc_pre}{q_pre}*<tool_call>)/\n\
             QWEN_TC_TO_FN: /{q_tc_pre}{q_pre}*<function=/\n\
             QWEN_TC_TO_CLOSE: /(?:<\\/tool_call>|{q_tc_pre}{q_pre}*<\\/tool_call>)/\n\
             QWEN_CLOSE_TO_FN: /(?:<function=|{q_pre}*<function=)/\n\
             QWEN_PARAMETER_OPEN: /<[ \\t\\r\\n]*parameter[ \\t\\r\\n]*=[ \\t\\r\\n]*/\n\
             QWEN_MARKUP: (/<[^<>]*>/ & ~(/<\\s*parameter\\s*=\\s*[^<>]*>/) & ~(/<\\/function>/) & ~(/<\\/tool_call>/) & ~(/<function=[^<>]*>/) & ~(/<tool_call>/))\n"
        ));
        let mut subgrammars = Vec::new();
        let has_structured_answer =
            self.prepare_answer_subgrammar(answer_constraint, true, &mut subgrammars)?;
        if !has_structured_answer && !plan.require_call {
            push_tagged_answer(
                &mut subgrammars,
                r"(?:<tool_call>|<\/tool_call>|<function=)",
            );
        }

        let wrap = if wrapping_newlines { "\"\\n\"" } else { "\"\"" };
        let close_param_str = if wrapping_newlines {
            "\n</parameter>"
        } else {
            "</parameter>"
        };
        let mut func_alts = Vec::new();
        let mut strict_names = Vec::new();

        for (t_idx, tool) in plan.tools.iter().enumerate() {
            if let Some(schema) = &tool.strict_schema {
                strict_names.push(tool.name.clone());
                let empty = Map::new();
                let props = schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty);
                let mut hints =
                    wire_hints(presentation.as_deref_mut(), &tool.name, !props.is_empty())?;
                let mut param_seq = Vec::with_capacity(props.len());
                for (p_idx, (prop_name, prop_schema)) in props.iter().enumerate() {
                    let split = split_top_property(schema, prop_schema);
                    if let SplitBranches::Pure {
                        string_branch,
                        nonstring_branch,
                    } = &split
                    {
                        for branch in [string_branch, nonstring_branch].into_iter().flatten() {
                            if branch != prop_schema {
                                budget.value(branch)?;
                            }
                        }
                    }
                    let open_lit =
                        serde_json::to_string(&format!("<parameter={prop_name}>")).unwrap();
                    let pname = format!("qwen_param_{t_idx}_{p_idx}");
                    match split {
                        SplitBranches::Pure {
                            string_branch: Some(sb),
                            nonstring_branch: None,
                        } => {
                            set_wire_hint(&mut hints, prop_name, prop_schema, true);
                            let s_sub = format!("qwen_s_{t_idx}_{p_idx}");
                            let g = build_raw_string_with_close_subgrammar(
                                &tool.name,
                                s_sub.clone(),
                                &sb,
                                ParameterClose::Text(close_param_str),
                                &["</parameter>", "<parameter=", "</function>", "</tool_call>"],
                            )?;
                            subgrammars.push(g);
                            rules.push_str(&format!("{pname}: {open_lit} {wrap} @{s_sub}\n"));
                        }
                        SplitBranches::Pure {
                            string_branch: None,
                            nonstring_branch: Some(nb),
                        } => {
                            set_wire_hint(&mut hints, prop_name, prop_schema, false);
                            let j_ref = push_json_param_subgrammar(
                                &mut subgrammars,
                                format!("qwen_j_{t_idx}_{p_idx}"),
                                schema,
                                &nb,
                            );
                            rules.push_str(&format!(
                                "{pname}: {open_lit} {wrap} {j_ref} {wrap} \"</parameter>\"\n"
                            ));
                        }
                        _ => {
                            return Err(grammar_err(format!(
                                "strict tool {:?} property {prop_name:?} mixes string and non-string types, which is ambiguous in function-tag tool syntax",
                                tool.name
                            )));
                        }
                    }
                    param_seq.push(pname);
                }
                let fname = format!("qwen_fn_{t_idx}");
                let fopen = serde_json::to_string(&format!("{}>", tool.name)).unwrap();
                if param_seq.is_empty() {
                    rules.push_str(&format!("{fname}: {fopen} ws \"</function>\"\n"));
                } else {
                    rules.push_str(&format!(
                        "{fname}: {fopen} ws {} ws \"</function>\"\n",
                        param_seq.join(" ws ")
                    ));
                }
                func_alts.push(fname);
            } else {
                let fopen = serde_json::to_string(&format!("{}>", tool.name)).unwrap();
                func_alts.push(format!(
                    "({fopen} ws QWEN_PARAMETER_OPEN @qwen_free_body) | ({fopen} ws \"</function>\")"
                ));
            }
        }

        if plan.tools.iter().any(|t| t.strict_schema.is_none()) || plan.allow_undeclared {
            let mut fb = GrammarWithLexer::from_lark(format!(
                "{}start: qwen_free_body\n",
                self.delimited_region(
                    "qwen_free_body",
                    "</function>",
                    &["</function>", "</tool_call>", "<tool_call>", "<function="]
                )
            ));
            fb.name = Some("qwen_free_body".into());
            subgrammars.push(fb);
        }
        if plan.allow_undeclared {
            let excl = strict_names
                .iter()
                .map(|n| serde_json::to_string(&format!("{n}>")).unwrap())
                .collect::<Vec<_>>()
                .join(" | ");
            let other = if excl.is_empty() {
                // Unreachable from a request (the compiler runs only with an active strict
                // tool), but an empty exclusion would emit `~()`, which is not a regex.
                "OTHER: /[^>\\s<]+>/\n".to_owned()
            } else {
                format!("OTHER: (/[^>\\s<]+>/ & ~({excl}))\n")
            };
            let mut og = GrammarWithLexer::from_lark(format!("{other}start: OTHER\n"));
            og.name = Some("qwen_other_name".into());
            subgrammars.push(og);
            func_alts.push("(@qwen_other_name /[ \\t\\r\\n]*/ QWEN_PARAMETER_OPEN @qwen_free_body) | (@qwen_other_name /[ \\t\\r\\n]*/ \"</function>\")".into());
        }

        rules.push_str(&format!(
            "qwen_fn_tail: {}\n\
             qwen_invocation: \"<function=\" qwen_fn_tail\n\
             tool_after_open: ws qwen_invocation ws \"</tool_call>\"\n\
             tool: \"<tool_call>\" tool_after_open | qwen_invocation\n",
            func_alts.join(" | ")
        ));
        let tail_tools = if plan.single_call { "" } else { " (ws tool)*" };
        rules.push_str(&format!("tools: tool{tail_tools}\n"));

        let tools_entry = if has_structured_answer {
            "tools".to_owned()
        } else {
            let pre_tc = GrammarWithLexer::from_lark(
                "PRE: (/(?s:.*)/ & ~(/(?s:.*)/ (\"<tool_call>\" | \"</tool_call>\" | \"<function=\") /(?s:.*)/))\n\
                 start[suffix=\"<tool_call>\"]: PRE\n"
                    .to_owned(),
            );
            let mut g_tc = pre_tc;
            g_tc.name = Some("qwen_pre_tc".into());
            subgrammars.push(g_tc);
            let pre_fn = GrammarWithLexer::from_lark(
                "PRE: (/(?s:.*)/ & ~(/(?s:.*)/ (\"<tool_call>\" | \"</tool_call>\" | \"<function=\") /(?s:.*)/))\n\
                 start[suffix=\"<function=\"]: PRE\n"
                    .to_owned(),
            );
            let mut g_fn = pre_fn;
            g_fn.name = Some("qwen_pre_fn".into());
            subgrammars.push(g_fn);
            let pre_close = GrammarWithLexer::from_lark(
                "PRE: (/(?s:.*)/ & ~(/(?s:.*)/ (\"<tool_call>\" | \"</tool_call>\" | \"<function=\") /(?s:.*)/))\n\
                 start[suffix=\"</tool_call>\"]: PRE\n"
                    .to_owned(),
            );
            let mut g_close = pre_close;
            g_close.name = Some("qwen_pre_close".into());
            subgrammars.push(g_close);
            rules.push_str(&format!(
                "tools_from_prose: @qwen_pre_tc tool_after_open{tail_tools} | @qwen_pre_fn qwen_fn_tail{tail_tools} | qwen_invocation{tail_tools}\n"
            ));
            "tools_from_prose".to_owned()
        };

        let body_branches = if plan.require_call {
            tools_entry
        } else if has_structured_answer {
            format!("(@answer | {tools_entry})")
        } else {
            let tc_prose = |tc_self: &str, fn_self: &str, close_self: &str, tail: &str| {
                format!(
                    "ws /[^<\\s]/ {tail} | ws QWEN_NONCALL_TOOL_LT {tail} | ws QWEN_TC_TO_TC {tc_self} | ws QWEN_TC_TO_FN {fn_self} | ws QWEN_TC_TO_CLOSE {close_self}"
                )
            };
            let fn_prose = |tc_self: &str, fn_self: &str, close_self: &str, tail: &str| {
                format!(
                    "QWEN_MENTION_TEXT {tail} | QWEN_MENTION_HEADER QWEN_MARKUP {tail} | QWEN_MENTION_SPACE QWEN_MARKUP {tail} | QWEN_MENTION_HEADER QWEN_NONCALL_FUNCTION_LT {tail} | QWEN_MENTION_SPACE QWEN_NONCALL_FUNCTION_LT {tail} | QWEN_MENTION_HEADER QWEN_FN_TO_TC {tc_self} | QWEN_MENTION_SPACE QWEN_FN_TO_TC {tc_self} | QWEN_MENTION_HEADER QWEN_FN_TO_FN {fn_self} | QWEN_MENTION_SPACE QWEN_FN_TO_FN {fn_self} | QWEN_MENTION_HEADER QWEN_FN_TO_CLOSE {close_self} | QWEN_MENTION_SPACE QWEN_FN_TO_CLOSE {close_self} | \">\""
                )
            };
            let close_prose = |tc_self: &str, fn_self: &str, close_self: &str, tail: &str| {
                format!(
                    "ws /[^<\\s]/ {tail} | ws QWEN_NONCALL_TOOL_LT {tail} | ws QWEN_TC_TO_TC {tc_self} | ws QWEN_CLOSE_TO_FN {fn_self} | ws QWEN_TC_TO_CLOSE {close_self} | /[ \\t\\r\\n]*/"
                )
            };
            let post_call_tail = if plan.single_call {
                "ws /[^<\\s]/ qwen_after | ws QWEN_NONCALL_TOOL_LT qwen_after | ws QWEN_TC_TO_TC qwen_after_tc | ws QWEN_CLOSE_TO_FN qwen_after_fn | ws QWEN_TC_TO_CLOSE qwen_after_close"
            } else {
                "ws /[^<\\s]/ qwen_body | ws QWEN_NONCALL_TOOL_LT qwen_body | ws QWEN_TC_TO_TC qwen_body_tc | ws QWEN_CLOSE_TO_FN qwen_body_fn | ws QWEN_TC_TO_CLOSE qwen_body_close"
            };
            rules.push_str(&format!(
                "qwen_post_call: {post_call_tail}\n\
                 qwen_body: @plain_answer | qwen_invocation{tail_tools} | qwen_invocation qwen_post_call | @qwen_pre_tc qwen_body_tc | @qwen_pre_fn qwen_body_fn | @qwen_pre_close qwen_body_close\n\
                 qwen_body_tc: tool_after_open{tail_tools} | tool_after_open qwen_post_call | {body_tc}\n\
                 qwen_body_fn: qwen_fn_tail{tail_tools} | qwen_fn_tail qwen_post_call | {body_fn}\n\
                 qwen_body_close: {body_close}\n\
                 qwen_after: @plain_answer | @qwen_pre_tc qwen_after_tc | @qwen_pre_fn qwen_after_fn | @qwen_pre_close qwen_after_close\n\
                 qwen_after_tc: {after_tc}\n\
                 qwen_after_fn: {after_fn}\n\
                 qwen_after_close: {after_close}\n",
                body_tc = tc_prose("qwen_body_tc", "qwen_body_fn", "qwen_body_close", "qwen_body"),
                body_fn = fn_prose("qwen_body_tc", "qwen_body_fn", "qwen_body_close", "qwen_body"),
                body_close = close_prose("qwen_body_tc", "qwen_body_fn", "qwen_body_close", "qwen_body"),
                after_tc = tc_prose("qwen_after_tc", "qwen_after_fn", "qwen_after_close", "qwen_after"),
                after_fn = fn_prose("qwen_after_tc", "qwen_after_fn", "qwen_after_close", "qwen_after"),
                after_close = close_prose("qwen_after_tc", "qwen_after_fn", "qwen_after_close", "qwen_after"),
            ));
            "qwen_body".to_owned()
        };

        if envelope.tool_body_start {
            rules.push_str(&format!("start: tool_after_open{tail_tools} ws\n"));
        } else if let Some(end) = envelope.reasoning_end {
            if envelope.tool_ends_reasoning {
                let mut rd = GrammarWithLexer::from_lark(format!(
                    "{}start: reasoning_done\n",
                    self.delimited_region("reasoning_done", end, &[end, "<tool_call>"])
                ));
                rd.name = Some("qwen_rdone".into());
                subgrammars.push(rd);
                let mut rt = GrammarWithLexer::from_lark(format!(
                    "{}start: reasoning_tool\n",
                    self.delimited_region("reasoning_tool", "<tool_call>", &[end, "<tool_call>"])
                ));
                rt.name = Some("qwen_rtool".into());
                subgrammars.push(rt);
                rules.push_str(&format!(
                    "start: (@qwen_rdone ws {body_branches} | @qwen_rtool tool_after_open{tail_tools}) ws\n"
                ));
            } else {
                let mut rg = GrammarWithLexer::from_lark(format!(
                    "{}start: reasoning\n",
                    self.region("reasoning", end)
                ));
                rg.name = Some("qwen_reasoning".into());
                subgrammars.push(rg);
                rules.push_str(&format!("start: @qwen_reasoning ws {body_branches} ws\n"));
            }
        } else {
            rules.push_str(&format!("start: ws {body_branches} ws\n"));
        }
        self.finish_composed(rules, subgrammars)
    }

    fn compile_strict_glm(
        &self,
        answer_constraint: Option<AnswerConstraint<'_>>,
        envelope: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        mut presentation: Option<&mut crate::output::tagged::OutputConfig>,
    ) -> Result<RequestGrammar> {
        let mut rules = String::from("GLM_WS1: /\\s+/\nws: GLM_WS1?\n");
        let mut subgrammars = Vec::new();
        let has_structured_answer =
            self.prepare_answer_subgrammar(answer_constraint, true, &mut subgrammars)?;
        if !has_structured_answer {
            push_tagged_answer(&mut subgrammars, r"<tool_call>");
        }

        let mut call_alts = Vec::new();
        let mut strict_names = Vec::new();

        for (t_idx, tool) in plan.tools.iter().enumerate() {
            if let Some(schema) = &tool.strict_schema {
                strict_names.push(tool.name.clone());
                let empty = Map::new();
                let props = schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty);
                let mut hints =
                    wire_hints(presentation.as_deref_mut(), &tool.name, !props.is_empty())?;
                let mut param_seq = Vec::with_capacity(props.len());
                for (p_idx, (prop_name, prop_schema)) in props.iter().enumerate() {
                    let open_lit = serde_json::to_string(&format!(
                        "<arg_key>{prop_name}</arg_key><arg_value>"
                    ))
                    .unwrap();
                    let pname = format!("glm_param_{t_idx}_{p_idx}");
                    match split_top_property(schema, prop_schema) {
                        SplitBranches::Pure {
                            string_branch: Some(sb),
                            nonstring_branch: None,
                        } => {
                            set_wire_hint(&mut hints, prop_name, prop_schema, true);
                            let s_sub = format!("glm_s_{t_idx}_{p_idx}");
                            let g = build_raw_string_with_close_subgrammar(
                                &tool.name,
                                s_sub.clone(),
                                &sb,
                                ParameterClose::Text("</arg_value>"),
                                &["</arg_value>", "</tool_call>"],
                            )?;
                            subgrammars.push(g);
                            rules.push_str(&format!("{pname}: {open_lit} @{s_sub}\n"));
                        }
                        SplitBranches::Pure {
                            string_branch: None,
                            nonstring_branch: Some(nb),
                        } => {
                            set_wire_hint(&mut hints, prop_name, prop_schema, false);
                            let j_ref = push_json_param_subgrammar(
                                &mut subgrammars,
                                format!("glm_j_{t_idx}_{p_idx}"),
                                schema,
                                &nb,
                            );
                            rules.push_str(&format!(
                                "{pname}: {open_lit} {j_ref} \"</arg_value>\"\n"
                            ));
                        }
                        _ => {
                            return Err(grammar_err(format!(
                                "strict tool {:?} property {prop_name:?} mixes string and non-string types, which is ambiguous in GLM argument-tag syntax",
                                tool.name
                            )));
                        }
                    }
                    param_seq.push(pname);
                }
                let cname = format!("glm_call_{t_idx}");
                let nlit = serde_json::to_string(&tool.name).unwrap();
                if param_seq.is_empty() {
                    rules.push_str(&format!("{cname}: ws {nlit} ws \"</tool_call>\"\n"));
                } else {
                    rules.push_str(&format!(
                        "{cname}: ws {nlit} ws {} ws \"</tool_call>\"\n",
                        param_seq.join(" ws ")
                    ));
                }
                call_alts.push(cname);
            } else {
                let nlit = serde_json::to_string(&tool.name).unwrap();
                call_alts.push(format!("(ws {nlit} @glm_free_tail)"));
            }
        }

        if plan.tools.iter().any(|t| t.strict_schema.is_none()) || plan.allow_undeclared {
            let mut fg = GrammarWithLexer::from_lark(
                "ws: /[ \\t\\r\\n]*/\n\
                 GLM_FREE_KEY: (/[^<]+/ & ~(/.*<\\/arg_key>.*/))\n\
                 GLM_FREE_BODY: (/(?s:.*)/ & ~(/(?s:.*)/ (\"</arg_value>\" | \"</tool_call>\") /(?s:.*)/))\n\
                 glm_free_val[suffix=\"</arg_value>\"]: GLM_FREE_BODY\n\
                 glm_free_arg: \"<arg_key>\" GLM_FREE_KEY \"</arg_key>\" ws \"<arg_value>\" glm_free_val\n\
                 start: (ws glm_free_arg)* ws \"</tool_call>\"\n"
                    .to_owned(),
            );
            fg.name = Some("glm_free_tail".into());
            subgrammars.push(fg);
        }
        let mut all_declared = Vec::new();
        for tool in &plan.tools {
            let lit = serde_json::to_string(&tool.name).unwrap();
            if !all_declared.contains(&lit) {
                all_declared.push(lit);
            }
        }
        if all_declared.is_empty() {
            rules.push_str("GLM_UNDECLARED_NAME: /[^<>\\s]+/\n");
        } else {
            rules.push_str(&format!(
                "GLM_UNDECLARED_NAME: (/[^<>\\s]+/ & ~({}))\n",
                all_declared.join(" | ")
            ));
        }
        if plan.allow_undeclared {
            call_alts.push("(ws GLM_UNDECLARED_NAME @glm_free_tail)".into());
        }

        rules.push_str(&format!(
            "tool_after_open: {}\n\
             tool: \"<tool_call>\" tool_after_open\n",
            call_alts.join(" | ")
        ));
        let tail_tools = if plan.single_call { "" } else { " (ws tool)*" };
        rules.push_str(&format!("tools: tool{tail_tools}\n"));

        let body_branches = if has_structured_answer {
            if plan.require_call {
                "tools".to_owned()
            } else {
                "(@answer | tools)".to_owned()
            }
        } else {
            let mut name_alts = all_declared;
            name_alts.push("GLM_UNDECLARED_NAME".into());
            let immed_pre = Self::proper_lt_prefixes(&["<tool_call>"]);
            let bad_pre = Self::proper_lt_prefixes(&["<arg_key>", "</tool_call>", "<tool_call>"]);
            let non_tc_lt = Self::non_marker_lt_regex(&["tool_call>"], "");
            let non_call_lt =
                Self::non_marker_lt_regex(&["arg_key>", "/tool_call>", "tool_call>"], "");
            rules.push_str(&format!(
                "glm_any_name: {names}\n\
                 GLM_PROSE_CHAR: /[^<>\\s]/\n\
                 GLM_IMMED_LT: /{immed_pre}*{non_tc_lt}/\n\
                 GLM_IMMED_PRE_TC: /{immed_pre}+<tool_call>/\n\
                 GLM_BAD_LT: /(?:{non_call_lt}|{bad_pre}{immed_pre}*{non_tc_lt})/\n\
                 GLM_BAD_PRE_TC: /{bad_pre}{immed_pre}*<tool_call>/\n",
                names = name_alts.join(" | "),
            ));

            let mut pg = GrammarWithLexer::from_lark(
                "PRE: (/(?s:.*)/ & ~(/(?s:.*)/ \"<tool_call>\" /(?s:.*)/))\n\
                 start[suffix=\"<tool_call>\"]: PRE\n"
                    .to_owned(),
            );
            pg.name = Some("glm_pre_tc".into());
            subgrammars.push(pg);

            let mention_step = |same_tc: &str, rest: &str| {
                format!(
                    "ws \">\" {rest} \
                     | ws GLM_IMMED_LT {rest} \
                     | ws GLM_IMMED_PRE_TC {same_tc} \
                     | ws \"<tool_call>\" {same_tc} \
                     | ws glm_any_name \">\" {rest} \
                     | ws glm_any_name GLM_BAD_LT {rest} \
                     | ws glm_any_name GLM_BAD_PRE_TC {same_tc} \
                     | ws glm_any_name \"<tool_call>\" {same_tc} \
                     | ws glm_any_name GLM_WS1 \">\" {rest} \
                     | ws glm_any_name GLM_WS1 GLM_PROSE_CHAR {rest} \
                     | ws glm_any_name GLM_WS1 GLM_BAD_LT {rest} \
                     | ws glm_any_name GLM_WS1 GLM_BAD_PRE_TC {same_tc} \
                     | ws glm_any_name GLM_WS1 \"<tool_call>\" {same_tc}"
                )
            };
            rules.push_str(&format!(
                "glm_after: @plain_answer | @glm_pre_tc glm_after_mention\n\
                 glm_after_mention: {}\n",
                mention_step("glm_after_mention", "glm_after"),
            ));
            let post_call = if plan.single_call {
                "(ws GLM_PROSE_CHAR glm_after | ws \">\" glm_after | ws GLM_IMMED_LT glm_after | ws GLM_IMMED_PRE_TC glm_after_mention | ws \"<tool_call>\" glm_after_mention)?"
            } else {
                "(ws GLM_PROSE_CHAR glm_body | ws \">\" glm_body | ws GLM_IMMED_LT glm_body | ws GLM_IMMED_PRE_TC glm_after_tc | ws \"<tool_call>\" glm_after_tc)?"
            };
            rules.push_str(&format!(
                "glm_call_seq: tool_after_open {post_call}\n\
                 glm_body: @plain_answer | @glm_pre_tc glm_after_tc\n\
                 glm_after_tc: glm_call_seq | {}\n",
                mention_step("glm_after_tc", "glm_body"),
            ));

            if plan.require_call {
                rules.push_str(&format!(
                    "glm_req_body: @glm_pre_tc glm_req_after_tc\n\
                     glm_req_after_tc: glm_call_seq | {}\n",
                    mention_step("glm_req_after_tc", "glm_req_body"),
                ));
                "glm_req_body".to_owned()
            } else {
                "glm_body".to_owned()
            }
        };

        if envelope.tool_body_start {
            rules.push_str(&format!("start: tool_after_open{tail_tools} ws\n"));
        } else if let Some(end) = envelope.reasoning_end {
            let mut rg = GrammarWithLexer::from_lark(format!(
                "{}start: reasoning\n",
                self.region("reasoning", end)
            ));
            rg.name = Some("glm_reasoning".into());
            subgrammars.push(rg);
            rules.push_str(&format!("start: @glm_reasoning ws {body_branches} ws\n"));
        } else {
            rules.push_str(&format!("start: ws {body_branches} ws\n"));
        }
        self.finish_composed(rules, subgrammars)
    }

    fn finish_composed(
        &self,
        rules: String,
        mut subgrammars: Vec<GrammarWithLexer>,
    ) -> Result<RequestGrammar> {
        let mut total_bytes = rules.len();
        for g in &subgrammars {
            total_bytes = total_bytes.saturating_add(
                serde_json::to_string(g)
                    .map(|s| s.len())
                    .unwrap_or(MAX_GRAMMAR_BYTES + 1),
            );
        }
        if total_bytes > MAX_GRAMMAR_BYTES {
            return Err(grammar_err(format!(
                "composed strict tool grammar exceeds {MAX_GRAMMAR_BYTES} bytes"
            )));
        }
        let mut grammars = Vec::with_capacity(1 + subgrammars.len());
        grammars.push(GrammarWithLexer::from_lark(rules));
        grammars.append(&mut subgrammars);
        let top = TopLevelGrammar {
            grammars,
            max_tokens: None,
        };
        self.compile(top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammar::tests::{byte_tokenizer, byte_tokenizer_with},
        model::contract::{ReasoningParser, ReasoningParserKind, ToolParser, ToolParserKind},
        output::{
            Event, Input, OutputConfig, OutputPipeline, Termination,
            tagged::{OutputConfig as TaggedConfig, OutputStart},
            tml::CONTROL_MARKERS,
        },
    };
    use serde_json::json;

    fn accepts_bytes(
        compiler: &GrammarCompiler,
        env: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        text: &str,
        eos: u32,
    ) -> bool {
        let mut g = compiler
            .compile_with_strict_tools(None, Some(env), plan)
            .unwrap();
        for &b in text.as_bytes() {
            if g.advance(u32::from(b)).is_err() {
                return false;
            }
        }
        g.advance(eos).is_ok()
    }

    #[test]
    fn strict_schema_admission_accepts_closed_subset_and_rejects_open_or_unsupported() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let valid = [
            json!({"type": "object", "additionalProperties": false}),
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "minLength": 2, "maxLength": 20, "pattern": "^[A-Za-z ]+$"},
                    "units": {"enum": ["c", "f"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                    "ratio": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0, "multipleOf": 0.25},
                    "note": {"type": ["string", "null"]},
                    "pair": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": false, "minItems": 2, "maxItems": 2},
                    "child": {"$ref": "#/$defs/Node"}
                },
                "required": ["city", "units", "days", "ratio", "note", "pair", "child"],
                "additionalProperties": false,
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "label": {"const": "leaf"},
                            "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}
                        },
                        "required": ["label", "next"],
                        "additionalProperties": false
                    }
                }
            }),
        ];
        for s in &valid {
            validate_strict_tool_schema(&compiler, "weather", s).unwrap();
        }

        let invalid = [
            json!({"type": "object"}),
            json!({"type": "object", "additionalProperties": true}),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "a"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"nested": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}, "required": ["nested"], "additionalProperties": false}),
            json!({"type": "object", "oneOf": [{"type": "object", "additionalProperties": false}], "additionalProperties": false}),
            json!({"type": "object", "allOf": [{"type": "object", "additionalProperties": false}], "additionalProperties": false}),
            json!({"type": "object", "not": {"type": "null"}, "additionalProperties": false}),
            json!({"type": "object", "patternProperties": {"^a": {"type": "string"}}, "additionalProperties": false}),
            json!({"type": "object", "minProperties": 1, "additionalProperties": false}),
            json!({"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "integer"}, "uniqueItems": true}}, "required": ["xs"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"x": {"$ref": "https://example.com/s.json"}}, "required": ["x"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/Loop"}}, "required": ["x"], "additionalProperties": false, "$defs": {"Loop": {"$ref": "#/$defs/Loop"}}}),
            json!({"type": "object", "properties": {"s": {"type": "string", "pattern": "[unclosed"}}, "required": ["s"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"n": {"type": "number", "multipleOf": 0.0}}, "required": ["n"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 1}}, "required": ["n"], "additionalProperties": false}),
        ];
        for s in &invalid {
            assert!(
                validate_strict_tool_schema(&compiler, "bad", s).is_err(),
                "expected strict rejection for {s}"
            );
        }
    }

    #[test]
    fn qwen_and_glm_strict_grammars_constrain_calls_and_parse_cleanly() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 2, "pattern": r"^(Asia\/Tokyo|Berlin)$"},
                "days": {"type": "integer"},
                "filter": {"anyOf":[{"$ref":"#/$defs/filter"},{"type":"null"}]}
            },
            "$defs":{"filter":{"type":"object","properties":{"hourly":{"type":["boolean","string"]}},"required":["hourly"],"additionalProperties":false}},
            "required": ["city", "days", "filter"],
            "additionalProperties": false
        });

        // QwenCoder compact (wrapping_newlines = 0) and newline-wrapped (wrapping_newlines = 1)
        for wrapping in [0usize, 1usize] {
            let mut syntax = crate::model::testing::metadata().frontend;
            syntax.reasoning = None;
            syntax.tools = Some(ToolParser {
                kind: ToolParserKind::QwenCoder,
                wrapping_newlines: wrapping as u32,
                tag_separator: String::new(),
            });
            let mut out_cfg = OutputConfig {
                syntax,
                control_tokens: vec![],
                content: TaggedConfig {
                    structured: false,
                    start: OutputStart::Content,
                    tools: vec![
                        json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                    ],
                },
            };
            let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
            let plan = ToolGrammarPlan {
                tools: vec![StrictToolSpec {
                    name: "weather".into(),
                    strict_schema: Some(&schema),
                }],
                syntax: Some((ToolParserKind::QwenCoder, wrapping, String::new())),
                dsml_token: None,
                allow_undeclared: true,
                require_call: false,
                single_call: true,
            };
            compiler
                .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
                .unwrap();
            let nl = if wrapping > 0 { "\n" } else { "" };
            let good_call = format!(
                "<tool_call><function=weather><parameter=city>{nl}Asia/Tokyo{nl}</parameter><parameter=days>{nl}3{nl}</parameter><parameter=filter>{nl}{{\"hourly\":true}}{nl}</parameter></function></tool_call>"
            );
            assert!(accepts_bytes(
                &compiler,
                &env,
                &plan,
                "Plain prose without tools.",
                256
            ));
            assert_tag_mention(&compiler, &env, &plan, out_cfg.clone());
            let bare = format!(
                "Checking. <function=weather><parameter=city>{nl}Asia/Tokyo{nl}</parameter><parameter=days>{nl}3{nl}</parameter><parameter=filter>{nl}{{\"hourly\":true}}{nl}</parameter></function>"
            );
            assert!(accepts_bytes(&compiler, &env, &plan, &bare, 256));
            assert_eq!(decoded_args(out_cfg.clone(), &bare)["city"], "Asia/Tokyo");
            assert!(accepts_bytes(&compiler, &env, &plan, &good_call, 256));
            assert!(!accepts_bytes(
                &compiler,
                &env,
                &plan,
                &format!("{good_call}\n{good_call}"),
                256
            ));
            assert!(!accepts_bytes(
                &compiler,
                &env,
                &plan,
                &format!("{good_call} and then {good_call}"),
                256
            ));
            let wrong_type = good_call.replace(
                &format!("<parameter=days>{nl}3{nl}"),
                &format!("<parameter=days>{nl}\"three\"{nl}"),
            );
            assert!(!accepts_bytes(&compiler, &env, &plan, &wrong_type, 256));
            for malformed_boundary in [
                good_call.replace(
                    &format!("<parameter=city>{nl}Asia/Tokyo{nl}</parameter>"),
                    &format!("<parameter=city>{nl}Asia</ parameter>Tokyo{nl}</parameter>"),
                ),
                good_call.replace(
                    &format!("<parameter=city>{nl}Asia/Tokyo{nl}</parameter>"),
                    &format!("<parameter=city>{nl}Asia< parameter =extra>Tokyo{nl}</parameter>"),
                ),
            ] {
                assert!(
                    !accepts_bytes(&compiler, &env, &plan, &malformed_boundary, 256),
                    "{malformed_boundary}"
                );
            }

            let nested = good_call.replace(
                "\"hourly\":true",
                "\"hourly\":\"literal </parameter></function></tool_call>\"",
            );
            assert!(accepts_bytes(&compiler, &env, &plan, &nested, 256));
            assert_eq!(
                decoded_args(out_cfg.clone(), &nested)["filter"]["hourly"],
                "literal </parameter></function></tool_call>"
            );
            let mut relaxed = out_cfg.clone();
            relaxed.content.tools[0]["function"]["strict"] = Value::Bool(false);
            let malformed = good_call.replace("{\"hourly\":true}", "{\"hourly\":\"unfinished");
            assert_eq!(
                decoded_args(relaxed, &malformed)["filter"],
                "{\"hourly\":\"unfinished"
            );
            let mut pipe = OutputPipeline::new(out_cfg);
            let mut evs = pipe.push(Input::Text(&good_call)).unwrap();
            evs.extend(pipe.finish(Termination::Eos).unwrap());
            let validator = validate_strict_tool_schema(&compiler, "weather", &schema).unwrap();
            let args: String = evs
                .iter()
                .filter_map(|e| match e {
                    Event::ToolArguments { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            validator.validate(&args).unwrap();
        }

        // GlmArgumentTags with required tool_choice
        let mut glm_syntax = crate::model::testing::metadata().frontend;
        glm_syntax.reasoning = None;
        glm_syntax.tools = Some(ToolParser {
            kind: ToolParserKind::GlmArgumentTags,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut glm_cfg = OutputConfig {
            syntax: glm_syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                ],
            },
        };
        let glm_env = OutputPipeline::grammar_envelope(&glm_cfg).unwrap();
        let glm_plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::GlmArgumentTags, 0, String::new())),
            dsml_token: None,
            allow_undeclared: false,
            require_call: true,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(
                None,
                Some(&glm_env),
                &glm_plan,
                Some(&mut glm_cfg.content),
            )
            .unwrap();
        let glm_good = "<tool_call>weather<arg_key>city</arg_key><arg_value>Berlin</arg_value><arg_key>days</arg_key><arg_value>5</arg_value><arg_key>filter</arg_key><arg_value>{\"hourly\":false}</arg_value></tool_call>";
        assert!(!accepts_bytes(
            &compiler,
            &glm_env,
            &glm_plan,
            "No tool call",
            256
        ));
        assert!(accepts_bytes(&compiler, &glm_env, &glm_plan, glm_good, 256));
        let nested = glm_good.replace(
            "\"hourly\":false",
            "\"hourly\":\"literal </arg_value></tool_call>\"",
        );
        assert!(accepts_bytes(&compiler, &glm_env, &glm_plan, &nested, 256));
        let glm_auto = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            require_call: false,
            allow_undeclared: true,
            ..glm_plan
        };
        assert_tag_mention(&compiler, &glm_env, &glm_auto, glm_cfg.clone());
        let mut relaxed = glm_cfg.clone();
        relaxed.content.tools[0]["function"]["strict"] = Value::Bool(false);
        let malformed = glm_good.replace("{\"hourly\":false}", "{\"hourly\":\"unfinished");
        assert_eq!(
            decoded_args(relaxed, &malformed)["filter"],
            "{\"hourly\":\"unfinished"
        );
        assert_eq!(
            decoded_args(glm_cfg, &nested)["filter"]["hourly"],
            "literal </arg_value></tool_call>"
        );
    }

    fn assert_tag_mention(
        c: &GrammarCompiler,
        env: &OutputEnvelope,
        plan: &ToolGrammarPlan,
        cfg: OutputConfig,
    ) {
        let text = "Use `<tool_call>` to mark a call.";
        assert!(accepts_bytes(c, env, plan, text, 256));
        let mut pipe = OutputPipeline::new(cfg);
        let mut events = pipe.push(Input::Text(text)).unwrap();
        events.extend(pipe.finish(Termination::Eos).unwrap());
        let mut content = String::new();
        for e in events {
            match e {
                Event::Content(s) => content.push_str(&s),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(content, text);
    }

    fn decoded_args(cfg: OutputConfig, wire: &str) -> Value {
        let mut parser = OutputPipeline::new(cfg);
        let mut args = String::new();
        for part in wire.as_bytes().chunks(1) {
            for event in parser
                .push(Input::Text(std::str::from_utf8(part).unwrap()))
                .unwrap()
            {
                if let Event::ToolArguments { text, .. } = event {
                    args.push_str(&text);
                }
            }
        }
        for event in parser.finish(Termination::Eos).unwrap() {
            if let Event::ToolArguments { text, .. } = event {
                args.push_str(&text);
            }
        }
        serde_json::from_str(&args).unwrap()
    }

    #[test]
    fn tml_json_strict_grammar_rebases_root_ref_and_parses_end_to_end() {
        let tokenizer = byte_tokenizer_with(&CONTROL_MARKERS);
        let controls: Vec<_> = CONTROL_MARKERS
            .iter()
            .map(|m| (tokenizer.token_to_id(m).unwrap(), (*m).to_owned()))
            .collect();
        let compiler = GrammarCompiler::new(&tokenizer, 320, &[256, 257], &controls).unwrap();
        let schema = json!({
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}
            },
            "required": ["label", "child"],
            "additionalProperties": false
        });
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::TmlJson,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ContentHeaders,
            open: "<|content_thinking|>".into(),
            close: "<|end_message|>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        let out_cfg = OutputConfig {
            syntax,
            control_tokens: controls.clone(),
            content: TaggedConfig {
                structured: false,
                start: OutputStart::RecipientHeader,
                tools: vec![
                    json!({"type":"function","function":{"name":"tree","parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "tree".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::TmlJson, 0, String::new())),
            dsml_token: None,
            allow_undeclared: false,
            require_call: true,
            single_call: true,
        };
        let invoke_id = tokenizer
            .token_to_id("<|content_invoke_tool_json|>")
            .unwrap();
        let end_id = tokenizer.token_to_id("<|end_message|>").unwrap();

        let advance_tml = |payload: &str| -> bool {
            let mut rg = compiler
                .compile_with_strict_tools(None, Some(&env), &plan)
                .unwrap();
            for &b in b"tree" {
                if rg.advance(u32::from(b)).is_err() {
                    return false;
                }
            }
            if rg.advance(invoke_id).is_err() {
                return false;
            }
            for &b in payload.as_bytes() {
                if rg.advance(u32::from(b)).is_err() {
                    return false;
                }
            }
            rg.advance(end_id).is_ok() && rg.advance(256).is_ok()
        };

        let payload =
            r#"{"name":"tree","args":{"label":"root","child":{"label":"leaf","child":null}}}"#;
        assert!(advance_tml(payload));
        let mut parser = OutputPipeline::new(out_cfg);
        let mut events = parser.push(Input::Text("tree")).unwrap();
        for (id, marker, body) in [
            (invoke_id, "<|content_invoke_tool_json|>", payload),
            (end_id, "<|end_message|>", ""),
        ] {
            events.extend(parser.push(Input::Token(id)).unwrap());
            events.extend(parser.push(Input::Text(marker)).unwrap());
            if !body.is_empty() {
                events.extend(parser.push(Input::Text(body)).unwrap());
            }
        }
        events.extend(parser.finish(Termination::Eos).unwrap());
        let [
            Event::ToolBegin { name, .. },
            Event::ToolArguments { text, .. },
            Event::ToolEnd { .. },
        ] = events.as_slice()
        else {
            panic!("{events:?}")
        };
        assert_eq!(name, "tree");
        assert_eq!(
            serde_json::from_str::<Value>(text).unwrap(),
            json!({"label":"root","child":{"label":"leaf","child":null}})
        );
        validate_strict_tool_schema(&compiler, "tree", &schema)
            .unwrap()
            .validate(text)
            .unwrap();
        assert!(!advance_tml(
            r#"{"name":"tree","args":{"label":"root","extra":1,"child":null}}"#
        ));
        let auto = ToolGrammarPlan {
            require_call: false,
            allow_undeclared: true,
            ..plan
        };
        let mut unknown = compiler
            .compile_with_strict_tools(None, Some(&env), &auto)
            .unwrap();
        assert!(
            !b"unknown"
                .iter()
                .all(|b| unknown.advance((*b).into()).is_ok())
                || unknown.advance(invoke_id).is_err()
        );
    }

    #[test]
    fn unsupported_dialects_reject_strict_tools_explicitly() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let env = OutputEnvelope {
            recipient: None,
            reasoning_end: None,
            tool: Some(("<tool>", "</tool>")),
            function: None,
            tool_ends_reasoning: false,
            response: None,
            message_close: None,
            tool_body_start: false,
        };
        let schema = json!({"type":"object","additionalProperties":false});
        for kind in [
            ToolParserKind::Xtml,
            ToolParserKind::MiniMaxXml,
            ToolParserKind::GemmaCall,
            ToolParserKind::Atem,
            ToolParserKind::InvokeParameterXml,
        ] {
            let plan = ToolGrammarPlan {
                tools: vec![StrictToolSpec {
                    name: "f".into(),
                    strict_schema: Some(&schema),
                }],
                syntax: Some((kind, 0, String::new())),
                dsml_token: None,
                allow_undeclared: false,
                require_call: false,
                single_call: false,
            };
            assert!(
                compiler
                    .compile_with_strict_tools(None, Some(&env), &plan)
                    .is_err(),
                "expected rejection for {kind:?}"
            );
        }
    }

    #[test]
    fn strict_literal_assertions_cannot_silently_drop_a_pattern() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","additionalProperties":false,"required":["color"],
            "properties":{"color":{"type":"string","enum":["red","blue"],"pattern":"^r"}}});
        assert!(validate_strict_tool_schema(&compiler, "paint", &schema).is_err());
    }

    #[test]
    fn embedded_tool_payload_keeps_reference_namespaces_and_aliases() {
        let schema = json!({"type":"object","additionalProperties":false,"required":["a","b"],
            "properties":{"a":{"$ref":"#/$defs/alias"},"b":{"$ref":"#/definitions/__strict_root__"}},
            "$defs":{"alias":{"$ref":"#/$defs/x~1y~0z"},"x/y~z":{"const":"hello"},"__strict_root__":{"const":9}},
            "definitions":{"__strict_root__":{"const":17}}});
        let SplitBranches::Pure {
            string_branch: Some(a),
            ..
        } = split_top_property(&schema, &schema["properties"]["a"])
        else {
            panic!("string alias lost")
        };
        assert_eq!(a["const"], "hello");
        let wrapper = embed_schema_in_tml_wrapper("f", &schema);
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        for (n, valid) in [(17, true), (9, false)] {
            let mut g = compiler.compile_json_schema(&wrapper).unwrap();
            let body = format!(r#"{{"name":"f","args":{{"a":"hello","b":{n}}}}}"#);
            let accepted =
                body.bytes().all(|b| g.advance(b.into()).is_ok()) && g.advance(256).is_ok();
            assert_eq!(accepted, valid, "{body}");
        }
    }

    #[test]
    fn strict_bounds_do_not_round_integers_or_fractional_multiples() {
        let schema = json!({"type":"object","additionalProperties":false,"required":["n","step"],
            "properties":{"n":{"type":"integer","maximum":9007199254740992u64},"step":{"type":"number","multipleOf":0.1}}});
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let exact = r#"{"n":9007199254740992,"step":0.3}"#;
        let rounded = r#"{"n":9007199254740993,"step":0.3}"#;
        let fractional = r#"{"n":1,"step":1e-12}"#;
        let grammar = validate_strict_tool_schema(&compiler, "measure", &schema).unwrap();
        grammar.validate(exact).unwrap();
        assert!(grammar.validate(rounded).is_err());
        assert!(grammar.validate(fractional).is_err());

        let schema = json!({"type":"object","properties":{"big":{"type":"integer","minimum":1}},
            "required":["big"],"additionalProperties":false});
        let grammar = validate_strict_tool_schema(&compiler, "measure", &schema).unwrap();
        grammar
            .validate(r#"{"big": 123456789012345678901234567890}"#)
            .unwrap();
        assert!(grammar.validate(r#"{"big": 0}"#).is_err());
    }

    #[test]
    fn strict_patterns_preserve_anchors_on_each_alternative() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","additionalProperties":false,"required":["v"],"properties":{"v":{"type":"string","pattern":"^a|b$"}}});
        let v = validate_strict_tool_schema(&c, "match", &schema).unwrap();
        for (s, valid) in [("ax", true), ("yb", true), ("xay", false)] {
            assert_eq!(
                v.validate(&json!({"v":s}).to_string()).is_ok(),
                valid,
                "{s}"
            );
        }
    }

    #[test]
    fn rejects_repeated_schema_expansion_before_building_subgrammars() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"one":{"$ref":"#/$defs/v"},"two":{"$ref":"#/$defs/v"}},
            "required":["one","two"],"additionalProperties":false,"$defs":{"v":{"type":"integer","description":"x".repeat(MAX_GRAMMAR_BYTES / 2)}}});
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "lookup".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: false,
            require_call: true,
            single_call: true,
        };
        let env = OutputEnvelope {
            recipient: None,
            reasoning_end: None,
            tool: Some(("<tool_call>", "</tool_call>")),
            function: None,
            tool_ends_reasoning: false,
            response: None,
            message_close: None,
            tool_body_start: false,
        };
        assert!(
            c.compile_with_strict_tools(None, Some(&env), &plan)
                .err()
                .unwrap()
                .to_string()
                .contains("schema expansion")
        );
        let mut dag = json!({"type":"object","properties":{"x":{"$ref":"#/$defs/r15"}},"required":["x"],"additionalProperties":false,"$defs":{"r0":{"type":"integer"}}});
        for i in 1..=15 {
            let r = json!({"$ref":format!("#/$defs/r{}",i-1)});
            dag["$defs"][format!("r{i}")] = json!({"anyOf":[r,r]});
        }
        let shared = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "f".into(),
                strict_schema: Some(&dag),
            }],
            ..plan
        };
        assert!(
            c.compile_with_strict_tools(None, Some(&env), &shared)
                .err()
                .unwrap()
                .to_string()
                .contains("schema expansion")
        );
    }

    #[test]
    fn tuple_tail_requires_a_schema_or_length_bound() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut s = json!({"type":"object","additionalProperties":false,"required":["xs"],"properties":{"xs":{"type":"array","prefixItems":[{"type":"integer"}]}}});
        assert!(validate_strict_tool_schema(&compiler, "f", &s).is_err());
        s["properties"]["xs"]["maxItems"] = json!(1);
        let v = validate_strict_tool_schema(&compiler, "f", &s).unwrap();
        assert!(v.validate(r#"{"xs":[1]}"#).is_ok());
        assert!(v.validate(r#"{"xs":[1,{}]}"#).is_err());
    }

    #[test]
    fn unguarded_reference_cycles_fail_before_compiler_expansion() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        // A is reachable unguarded through the direct branch of anyOf, and guarded through
        // property `b`. Adjacent branches, so branch order cannot change the verdict.
        let guarded = json!({"type":"object","properties":{"b":{"$ref":"#/$defs/B"}},"required":["b"],"additionalProperties":false});
        let unguarded = json!({"$ref":"#/$defs/B"});
        for guarded_first in [false, true] {
            let branches = if guarded_first {
                json!([guarded.clone(), unguarded.clone()])
            } else {
                json!([unguarded.clone(), guarded.clone()])
            };
            let schema = json!({"type":"object","properties":{"a":{"$ref":"#/$defs/A"}},"required":["a"],"additionalProperties":false,
                "$defs":{"A":{"anyOf":branches},"B":{"$ref":"#/$defs/A"}}});
            match validate_strict_tool_schema(&compiler, "f", &schema) {
                Ok(_) => panic!("guarded_first={guarded_first}: unguarded cycle admitted"),
                Err(error) => {
                    let text = error.to_string();
                    assert!(
                        text.contains("cyclic unguarded $ref"),
                        "guarded_first={guarded_first}: wrong rejection: {text}"
                    );
                }
            }
            let plan = ToolGrammarPlan {
                tools: vec![StrictToolSpec {
                    name: "f".into(),
                    strict_schema: Some(&schema),
                }],
                syntax: None,
                dsml_token: None,
                allow_undeclared: false,
                require_call: true,
                single_call: true,
            };
            let direct = compiler
                .compile_with_strict_tools(None, None, &plan)
                .err()
                .expect("direct strict compiler entry admitted an unguarded cycle")
                .to_string();
            assert!(
                direct.contains("cyclic unguarded $ref"),
                "guarded_first={guarded_first}: direct compiler returned {direct}"
            );
        }
        // Control: the same recursion reached only through object properties is admitted,
        // so the rejection above is the cycle walk, not an anyOf shape complaint.
        let guarded_only = json!({"type":"object","properties":{"a":{"$ref":"#/$defs/A"}},"required":["a"],"additionalProperties":false,
            "$defs":{"A":{"anyOf":[guarded.clone()]},"B":{"$ref":"#/$defs/A"}}});
        validate_strict_tool_schema(&compiler, "f", &guarded_only)
            .expect("guarded recursion must be admitted");
    }

    #[test]
    fn qwen_undeclared_name_rule_compiles_without_strict_tools() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![json!({"type":"function","function":{"name":"free"}})],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "free".into(),
                strict_schema: None,
            }],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            "<function=free></function>",
            256
        ));
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            "<function=other><parameter=x>1</parameter></function>",
            256
        ));
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            "prose <function=free> text",
            256
        ));
    }

    #[test]
    fn strict_large_flat_array_validates() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"xs":{"type":"array","items":{"type":"integer"}}},"required":["xs"],"additionalProperties":false});
        let v = validate_strict_tool_schema(&c, "f", &schema).unwrap();
        let xs: Vec<i64> = (0..20_000).collect();
        let arg = json!({"xs": xs}).to_string();
        match v.validate(&arg) {
            Ok(()) => {}
            Err(e) => panic!("{} bytes: {e}", arg.len()),
        }
    }

    #[test]
    fn strict_stacked_references_accept_a_trailing_variant() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut defs = serde_json::Map::new();
        defs.insert("a0".into(), json!({"type":"integer"}));
        for i in 1..=24 {
            defs.insert(format!("a{i}"), json!({"$ref":format!("#/$defs/a{}", i-1)}));
        }
        let schema = json!({"type":"object","properties":{"v":{"anyOf":[{"type":"string"},{"$ref":"#/$defs/a24"}]}},"required":["v"],"additionalProperties":false,"$defs":defs});
        let v = validate_strict_tool_schema(&c, "f", &schema).unwrap();
        assert!(v.validate(r#"{"v":7}"#).is_ok());
    }

    #[test]
    fn strict_recursive_union_accepts_a_deep_value() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"root":{"$ref":"#/$defs/E"}},"required":["root"],"additionalProperties":false,
        "$defs":{"E":{"anyOf":[
            {"type":"null"},
            {"type":"object","properties":{"payload":{"$ref":"#/$defs/E"},"kind":{"const":"a"}},"required":["payload","kind"],"additionalProperties":false},
            {"type":"object","properties":{"payload":{"$ref":"#/$defs/E"},"kind":{"const":"b"}},"required":["payload","kind"],"additionalProperties":false}
        ]}}});
        let v = validate_strict_tool_schema(&c, "f", &schema).unwrap();
        let mut value = json!(null);
        for _ in 0..10 {
            value = json!({"payload": value, "kind": "b"});
        }
        let arg = json!({"root": value}).to_string();
        match v.validate(&arg) {
            Ok(()) => {}
            Err(e) => panic!("{} bytes: {e}", arg.len()),
        }
    }

    #[test]
    fn strict_array_of_unions_validates() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"xs":{"type":"array","items":{"anyOf":[{"$ref":"#/$defs/U"},{"type":"null"}]}}},"required":["xs"],"additionalProperties":false,
            "$defs":{"U":{"anyOf":[{"type":"string"},{"type":"integer"}]}}});
        let v = validate_strict_tool_schema(&c, "f", &schema).unwrap();
        let xs: Vec<Value> = (0..5_000)
            .map(|i| if i % 2 == 0 { json!(i) } else { Value::Null })
            .collect();
        let arg = json!({"xs": xs}).to_string();
        match v.validate(&arg) {
            Ok(()) => {}
            Err(e) => panic!("{} bytes: {e}", arg.len()),
        }
        assert!(v.validate(&json!({"xs":[true]}).to_string()).is_err());
    }

    #[test]
    fn qwen_parallel_calls_allow_prose_between() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: false,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        let call = "<function=weather><parameter=city>Asia/Tokyo</parameter></function>";
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("{call} then {call}"),
            256
        ));
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            "<function=weather></function>",
            256
        ));
    }

    fn assert_prose_at_eos(config: &OutputConfig, text: &str) {
        use crate::output::{Event, Input, Termination};
        let mut parser = OutputPipeline::new(config.clone());
        let mut events = parser.push(Input::Text(text)).unwrap();
        events.extend(parser.finish(Termination::Eos).unwrap());
        assert_eq!(events, [Event::Content(text.into())], "{text:?}");
    }

    fn assert_unfinished_at_eos(config: &OutputConfig, text: &str) {
        use crate::output::{Input, Termination};
        let mut parser = OutputPipeline::new(config.clone());
        parser.push(Input::Text(text)).unwrap();
        assert!(parser.finish(Termination::Eos).is_err(), "{text:?}");
    }

    #[test]
    fn qwen_tag_mentions_work_with_wrapped_parameters() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 1,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::QwenCoder, 1, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        for text in [
            "Use <function=weather> to mark a call.",
            "Avoid <tool_call>.",
            "Close with </tool_call>.",
        ] {
            assert!(accepts_bytes(&compiler, &env, &plan, text, 256), "{text}");
            assert_prose_at_eos(&out_cfg, text);
        }
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            "Avoid <tool_call>",
            256
        ));
        assert_unfinished_at_eos(&out_cfg, "Avoid <tool_call>");
    }

    #[test]
    fn qwen_string_values_round_trip_tool_markers() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","strict":true,"parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: false,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        for marker in ["a<function=x>b", "a<tool_call>b"] {
            let wire = format!("<function=weather><parameter=city>{marker}</parameter></function>");
            assert!(
                accepts_bytes(&compiler, &env, &plan, &wire, 256),
                "{marker}"
            );
            assert_eq!(decoded_args(out_cfg.clone(), &wire)["city"], marker);
        }
    }

    #[test]
    fn qwen_prose_may_mention_the_function_tag() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "weather".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        for text in [
            "Use <function=weather> to mark a call.",
            "Tags look like <function=weather> and <parameter=city>.",
            "Mention <function=weather> only.",
            "Use <parameterized> as an example.",
            "Use <function=weather> <parameterized> as prose.",
            "<function=>",
            "Use <function=weather extra> literally.",
            "Use <function=weather> <parameter name=q> literally.",
            "Use <function=weather> <b>here</b>.",
            "Use <function=weather> <3",
            "Use <tool_call><3",
            "Use <tool_call><foo> literally.",
            "Use <tool_call><tool_<b> literally.",
            "Use <tool_call><<3 in prose.",
            "Use <function=weather><<3 in prose.",
            "Use <function=weather><tool_<b> in prose.",
            "Use <function=weather></ function> x",
            "Use <function=weather></\nfunction> x",
            "Use `<tool_call>` to mark a call.",
            "Close with </tool_call>.",
            "Avoid </tool_call>",
            "Use <tool_call></tool_call><b> literally.",
            "Use </tool_call><b> literally.",
        ] {
            assert!(accepts_bytes(&compiler, &env, &plan, text, 256), "{text}");
            assert_prose_at_eos(&out_cfg, text);
        }
        // Backticks and chained `<` before `<tool_call>` do not escape a complete native invocation.
        for inline_call in [
            "Use `<tool_call><function=weather><parameter=city>NYC</parameter></function></tool_call>` to call.",
            "Use <tool_call><<tool_call><function=weather><parameter=city>NYC</parameter></function></tool_call>",
            "Use <function=weather><tool_call><function=weather><parameter=city>NYC</parameter></function></tool_call>",
            "Use <function=weather><function=weather><parameter=city>NYC</parameter></function>",
        ] {
            assert!(
                accepts_bytes(&compiler, &env, &plan, inline_call, 256),
                "{inline_call}"
            );
            assert_eq!(decoded_args(out_cfg.clone(), inline_call)["city"], "NYC");
        }
        for text in [
            "Avoid <tool_call>",
            "Avoid <tool_call>   ",
            "Avoid <tool_call><<tool_call>",
            "Avoid <function=weather><<tool_call>",
        ] {
            assert_unfinished_at_eos(&out_cfg, text);
        }
        for padded in [
            "Avoid <tool_call>",
            "Avoid <tool_call>   ",
            "Avoid <tool_call><<tool_call>",
            "Avoid <function=weather><<tool_call>",
            "<function=weather>< parameter =city>Paris</parameter></function>",
            "<function=weather><parameter = city>Paris</parameter></function>",
            "Avoid <function=weather>",
            "Avoid <function=weather>   ",
        ] {
            assert!(
                !accepts_bytes(&compiler, &env, &plan, padded, 256),
                "{padded}"
            );
        }
        let call = "<function=weather><parameter=city>Asia/Tokyo</parameter></function>";
        assert!(accepts_bytes(&compiler, &env, &plan, call, 256));
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("Use <function=weather> then call {call}"),
            256
        ));
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("{call}\n{call}"),
            256
        ));
        assert!(accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("First {call} then stop."),
            256
        ));
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("First {call} then {call}"),
            256
        ));
        let wrong = "<function=weather></function>";
        assert!(!accepts_bytes(&compiler, &env, &plan, wrong, 256));
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            &format!("Use <function=weather> then call {wrong}"),
            256
        ));
    }

    #[test]
    fn glm_prose_may_mention_the_tool_call_tag() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let zero_schema =
            json!({"type":"object","properties":{},"required":[],"additionalProperties":false});
        let city_schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::GlmArgumentTags,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"weather","strict":true,"parameters":zero_schema}}),
                    json!({"type":"function","function":{"name":"lookup","strict":true,"parameters":city_schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![
                StrictToolSpec {
                    name: "weather".into(),
                    strict_schema: Some(&zero_schema),
                },
                StrictToolSpec {
                    name: "lookup".into(),
                    strict_schema: Some(&city_schema),
                },
            ],
            syntax: Some((ToolParserKind::GlmArgumentTags, 0, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();

        for text in [
            "Use `<tool_call>` to mark a call.",
            "Mention <tool_call> in prose.",
            "Use <tool_call>weather literally.",
            "Use <tool_call><3",
            "Avoid </tool_call>",
            "Mention <tool_call>>",
            "Mention <tool_call>weather>",
            "Mention <tool_call>weather <b>",
            "Mention <tool_call>weather<arg_keys>",
            "Mention <tool_call>weather</tool_calls>",
            "Mention <tool_call><tool_call> in prose.",
            "Mention <tool_call><<3 in prose.",
            "Mention <tool_call></tool_cal",
            "Mention <tool_call>   </tool_cal",
            "Mention <tool_call><arg_k",
            "Mention <tool_call>   <arg_k",
            "Mention <tool_call>weather <tool_call> in prose.",
            "Mention <tool_call>weather<arg<3 in prose.",
        ] {
            assert!(accepts_bytes(&compiler, &env, &plan, text, 256), "{text}");
            assert_prose_at_eos(&out_cfg, text);
        }

        for text in [
            "Avoid <tool_call>",
            "Avoid <tool_call>   ",
            "Avoid <tool_call>\u{a0}",
            "Avoid <tool_call>weather",
            "Avoid <tool_call>weather   ",
            "Avoid `<tool_call>`",
            "Avoid <tool_call><tool_call>",
            "Avoid <tool_call><<tool_call>",
            "Avoid <tool_call><tool_<tool_call>",
            "Avoid <tool_call>weather <tool_call>",
            "Avoid <tool_call>weather<<tool_call>",
            "Avoid <tool_call>weather<arg<tool_call>",
            "Mention <tool_call> in prose; avoid <tool_call>weather",
        ] {
            assert!(!accepts_bytes(&compiler, &env, &plan, text, 256), "{text}");
            assert_unfinished_at_eos(&out_cfg, text);
        }

        for tag in ["<arg_key>", "</tool_call>"] {
            for end in 1..tag.len() {
                for gap in ["", " \t", "\u{a0}"] {
                    let text = format!("Avoid <tool_call>weather{gap}{}", &tag[..end]);
                    assert!(!accepts_bytes(&compiler, &env, &plan, &text, 256), "{text}");
                    assert_unfinished_at_eos(&out_cfg, &text);
                }
            }
        }

        for inline_call in [
            "Use `<tool_call>weather</tool_call>` to call.".to_owned(),
            "Mention <tool_call><tool_call>weather</tool_call>".to_owned(),
        ]
        .into_iter()
        .chain((1.."<tool_call>".len()).map(|end| {
            format!(
                "Mention <tool_call>{}<tool_call>weather</tool_call>",
                &"<tool_call>"[..end]
            )
        })) {
            assert!(
                accepts_bytes(&compiler, &env, &plan, &inline_call, 256),
                "{inline_call}"
            );
            assert_eq!(decoded_args(out_cfg.clone(), &inline_call), json!({}));
        }
        let arg_call = "Mention <tool_call> first, then <tool_call>lookup<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call> done.";
        assert!(accepts_bytes(&compiler, &env, &plan, arg_call, 256));
        assert_eq!(decoded_args(out_cfg.clone(), arg_call)["city"], "Tokyo");
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &plan,
            "<tool_call>weather</tool_call>\n<tool_call>weather</tool_call>",
            256,
        ));
    }

    #[test]
    fn qwen_and_glm_strict_parsers_preserve_30_digit_integers() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let big = "123456789012345678901234567890";
        let big_schema = json!({
            "type": "object",
            "$defs": {"BigInt": {"type": "integer"}},
            "properties": {
                "top": {"type": "integer"},
                "via_any": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "via_ref": {"$ref": "#/$defs/BigInt"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "arr": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["id", "arr"],
                    "additionalProperties": false
                }
            },
            "required": ["top", "via_any", "via_ref", "nested"],
            "additionalProperties": false
        });
        let validator = validate_strict_tool_schema(&compiler, "audit", &big_schema).unwrap();
        for (kind, wire) in [
            (
                ToolParserKind::QwenCoder,
                format!(
                    "<function=audit><parameter=top>{big}</parameter><parameter=via_any>{big}</parameter><parameter=via_ref>{big}</parameter><parameter=nested>{{\"id\":{big},\"arr\":[{big}]}}</parameter></function>"
                ),
            ),
            (
                ToolParserKind::GlmArgumentTags,
                format!(
                    "<tool_call>audit<arg_key>top</arg_key><arg_value>{big}</arg_value><arg_key>via_any</arg_key><arg_value>{big}</arg_value><arg_key>via_ref</arg_key><arg_value>{big}</arg_value><arg_key>nested</arg_key><arg_value>{{\"id\": {big}, \"arr\": [{big}]}}</arg_value></tool_call>"
                ),
            ),
        ] {
            let mut syn = crate::model::testing::metadata().frontend;
            syn.reasoning = None;
            syn.tools = Some(ToolParser {
                kind,
                wrapping_newlines: 0,
                tag_separator: String::new(),
            });
            let mut cfg = OutputConfig {
                syntax: syn,
                control_tokens: vec![],
                content: TaggedConfig {
                    structured: false,
                    start: OutputStart::Content,
                    tools: vec![json!({
                        "type": "function",
                        "function": {"name": "audit", "strict": true, "parameters": big_schema}
                    })],
                },
            };
            let e = OutputPipeline::grammar_envelope(&cfg).unwrap();
            let p = ToolGrammarPlan {
                tools: vec![StrictToolSpec {
                    name: "audit".into(),
                    strict_schema: Some(&big_schema),
                }],
                syntax: Some((kind, 0, String::new())),
                dsml_token: None,
                allow_undeclared: false,
                require_call: true,
                single_call: true,
            };
            compiler
                .compile_presented_strict_tools(None, Some(&e), &p, Some(&mut cfg.content))
                .unwrap();
            assert!(accepts_bytes(&compiler, &e, &p, &wire, 256), "{kind:?}");
            let mut pipe = OutputPipeline::new(cfg);
            let mut evs = pipe.push(crate::output::Input::Text(&wire)).unwrap();
            evs.extend(pipe.finish(crate::output::Termination::Eos).unwrap());
            let raw_args = evs
                .into_iter()
                .find_map(|ev| match ev {
                    crate::output::Event::ToolArguments { text, .. } => Some(text),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("expected tool arguments for {kind:?}"));
            assert!(
                validator.validate(&raw_args).is_ok(),
                "{kind:?}: {raw_args}"
            );
            assert_eq!(
                raw_args.matches(big).count(),
                5,
                "{kind:?} lost 30-digit precision: {raw_args}"
            );
        }
    }

    #[test]
    fn glm_required_parallel_grammar_rejects_prose_and_accepts_multiple_calls() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let zero_schema =
            json!({"type":"object","properties":{},"required":[],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::GlmArgumentTags,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({
                        "type": "function",
                        "function": {"name": "weather", "strict": true, "parameters": zero_schema}
                    }),
                    json!({"type": "function", "function": {"name": "free"}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let req_par_plan = ToolGrammarPlan {
            tools: vec![
                StrictToolSpec {
                    name: "weather".into(),
                    strict_schema: Some(&zero_schema),
                },
                StrictToolSpec {
                    name: "free".into(),
                    strict_schema: None,
                },
            ],
            syntax: Some((ToolParserKind::GlmArgumentTags, 0, String::new())),
            dsml_token: None,
            allow_undeclared: false,
            require_call: true,
            single_call: false,
        };
        compiler
            .compile_presented_strict_tools(
                None,
                Some(&env),
                &req_par_plan,
                Some(&mut out_cfg.content),
            )
            .unwrap();
        assert!(!accepts_bytes(
            &compiler,
            &env,
            &req_par_plan,
            "Mention <tool_call> in prose.",
            256
        ));
        assert!(accepts_bytes(
            &compiler,
            &env,
            &req_par_plan,
            "<tool_call>weather</tool_call>\n<tool_call>free<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            256
        ));
    }

    #[test]
    fn strict_const_and_enum_enforce_exact_integer_range() {
        let c = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let large: Value = serde_json::from_slice(b"1234567890123456789012345").unwrap();
        assert!(large.is_f64());
        assert!(unsupported_strict_literal(&large));
        let rounded: Value = serde_json::from_slice(b"9007199254740993.0").unwrap();
        assert_eq!(rounded.as_f64(), Some(9_007_199_254_740_992.0));
        assert!(unsupported_strict_literal(&rounded));

        for raw_lit in [
            "9007199254740992",
            "-9007199254740992",
            "9007199254740993",
            "-9007199254740993",
            "18446744073709551615",
            "1234567890123456789012345",
            "9007199254740993.0",
            "1e20",
        ] {
            for ty in [None, Some("\"integer\""), Some("\"number\"")] {
                let ty_kv = ty.map(|t| format!("\"type\":{t},")).unwrap_or_default();
                let const_doc: Value = serde_json::from_str(&format!(
                    r#"{{"type":"object","properties":{{"n":{{{ty_kv}"const":{raw_lit}}}}},"required":["n"],"additionalProperties":false}}"#
                ))
                .unwrap();
                let err = validate_strict_tool_schema(&c, "f", &const_doc)
                    .err()
                    .unwrap_or_else(|| panic!("admitted const {raw_lit} with {ty:?}"))
                    .to_string();
                assert!(
                    err.contains("exact 53-bit") || err.contains("outside its declared type"),
                    "{raw_lit} ({ty:?}): {err}"
                );

                let enum_doc: Value = serde_json::from_str(&format!(
                    r#"{{"type":"object","properties":{{"n":{{{ty_kv}"enum":[1,{raw_lit}]}}}},"required":["n"],"additionalProperties":false}}"#
                ))
                .unwrap();
                let err = validate_strict_tool_schema(&c, "f", &enum_doc)
                    .err()
                    .unwrap_or_else(|| panic!("admitted enum {raw_lit} with {ty:?}"))
                    .to_string();
                assert!(
                    err.contains("exact 53-bit") || err.contains("outside its declared type"),
                    "{raw_lit} ({ty:?}): {err}"
                );
            }
        }

        // Endpoint gate at ±(2^53 - 1): accepted at admission, accepts exact value, rejects neighbours.
        for (exact, prev, next) in [
            ("9007199254740991", "9007199254740990", "9007199254740992"),
            (
                "-9007199254740991",
                "-9007199254740992",
                "-9007199254740990",
            ),
        ] {
            for prop in [
                format!(r#"{{"type":"integer","const":{exact}}}"#),
                format!(r#"{{"type":"integer","enum":[{exact}]}}"#),
            ] {
                let doc: Value = serde_json::from_str(&format!(
                    r#"{{"type":"object","properties":{{"n":{prop}}},"required":["n"],"additionalProperties":false}}"#
                ))
                .unwrap();
                let g = validate_strict_tool_schema(&c, "f", &doc).unwrap();
                assert!(g.validate(&format!(r#"{{"n":{exact}}}"#)).is_ok(), "{prop}");
                assert!(
                    g.validate(&format!(r#"{{"n":{prev}}}"#)).is_err(),
                    "{prop} prev"
                );
                assert!(
                    g.validate(&format!(r#"{{"n":{next}}}"#)).is_err(),
                    "{prop} next"
                );
            }
        }

        // Small decimal and string-encoded long IDs remain valid.
        let dec_doc = json!({"type":"object","properties":{"x":{"type":"number","const":1.25},"id":{"type":"string","const":"1234567890123456789012345"}},"required":["x","id"],"additionalProperties":false});
        let g = validate_strict_tool_schema(&c, "f", &dec_doc).unwrap();
        assert!(
            g.validate(r#"{"x":1.25,"id":"1234567890123456789012345"}"#)
                .is_ok()
        );
    }

    #[test]
    fn qwen_free_body_cannot_cross_a_tool_boundary() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let mut out_cfg = OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"free"}}),
                    json!({"type":"function","function":{"name":"weather","strict":true,"parameters":schema}}),
                ],
            },
        };
        let env = OutputPipeline::grammar_envelope(&out_cfg).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![
                StrictToolSpec {
                    name: "free".into(),
                    strict_schema: None,
                },
                StrictToolSpec {
                    name: "weather".into(),
                    strict_schema: Some(&schema),
                },
            ],
            syntax: Some((ToolParserKind::QwenCoder, 0, String::new())),
            dsml_token: None,
            allow_undeclared: true,
            require_call: false,
            single_call: true,
        };
        compiler
            .compile_presented_strict_tools(None, Some(&env), &plan, Some(&mut out_cfg.content))
            .unwrap();
        for (text, want) in [
            ("<function=free><parameter=x>1</parameter></function>", true),
            ("<function=free>plain body</function>", true),
            (
                "<function=free>< parameter =x>1</ parameter></function>",
                true,
            ),
            (
                "<function=other>< parameter =x>1</ parameter></function>",
                true,
            ),
            (
                "<function=free>a</tool_call><tool_call><function=weather><parameter=city>NYC</parameter></function>",
                false,
            ),
            ("<tool_call><function=free>a</function></tool_call>", false),
            (
                "<tool_call><function=free><parameter=x>1</parameter></function></tool_call>",
                true,
            ),
            (
                "<function=weather><parameter=city>NYC</parameter></function>",
                true,
            ),
            (
                "<function=free><parameter=x>1<function=weather><parameter=city>NYC</parameter></function></parameter></function>",
                false,
            ),
            (
                "<function=free><tool_call><function=weather><parameter=city>NYC</parameter></function>",
                false,
            ),
        ] {
            let got = accepts_bytes(&compiler, &env, &plan, text, 256);
            assert_eq!(got, want, "{text}");
        }
        let padded = "<function=free>< parameter =x>1</ parameter></function>";
        assert_eq!(decoded_args(out_cfg, padded)["x"], "1");
    }

    #[test]
    fn strict_wire_grammars_reject_schema_violations_during_generation() {
        #[derive(Clone, Copy)]
        enum Constraint {
            MinLength,
            Pattern,
            Enum,
            NumericRange,
        }

        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            kind: ToolParserKind,
            separator: &'static str,
            start: OutputStart,
            controls: &'static [&'static str],
            valid: &'static str,
            invalid: &'static str,
            constraint: Constraint,
        }

        let cases = [
            Case {
                name: "Qwen",
                kind: ToolParserKind::QwenCoder,
                separator: "",
                start: OutputStart::Content,
                controls: &[],
                constraint: Constraint::MinLength,
                valid: "<function=weather><parameter=value>ok</parameter></function>",
                invalid: "<function=weather><parameter=value>x</parameter></function>",
            },
            Case {
                name: "GLM",
                kind: ToolParserKind::GlmArgumentTags,
                separator: "",
                start: OutputStart::Content,
                controls: &[],
                constraint: Constraint::Pattern,
                valid: "<tool_call>weather<arg_key>value</arg_key><arg_value>OK</arg_value></tool_call>",
                invalid: "<tool_call>weather<arg_key>value</arg_key><arg_value>no</arg_value></tool_call>",
            },
            Case {
                name: "DSML",
                kind: ToolParserKind::Dsml,
                separator: "",
                start: OutputStart::Content,
                controls: &["｜DSML｜"],
                constraint: Constraint::Enum,
                valid: "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"value\" string=\"true\">allowed</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
                invalid: "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"value\" string=\"true\">denied</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
            },
            Case {
                name: "TML",
                kind: ToolParserKind::TmlJson,
                separator: "",
                start: OutputStart::RecipientHeader,
                controls: &CONTROL_MARKERS,
                constraint: Constraint::NumericRange,
                valid: "weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"value\":2}}<|end_message|>",
                invalid: "weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"value\":4}}<|end_message|>",
            },
        ];

        for case in cases {
            let value_schema = match case.constraint {
                Constraint::MinLength => json!({"type": "string", "minLength": 2}),
                Constraint::Pattern => json!({"type": "string", "pattern": "^[A-Z]{2}$"}),
                Constraint::Enum => json!({"enum": ["allowed"]}),
                Constraint::NumericRange => {
                    json!({"type": "integer", "minimum": 1, "maximum": 3})
                }
            };
            let schema = json!({
                "type": "object",
                "properties": {"value": value_schema},
                "required": ["value"],
                "additionalProperties": false
            });
            let tokenizer = byte_tokenizer_with(case.controls);
            let controls: Vec<_> = case
                .controls
                .iter()
                .map(|marker| (tokenizer.token_to_id(marker).unwrap(), (*marker).to_owned()))
                .collect();
            let compiler = GrammarCompiler::new(&tokenizer, 320, &[256, 257], &controls).unwrap();
            let _validator = validate_strict_tool_schema(&compiler, "weather", &schema).unwrap();
            let mut syntax = crate::model::testing::metadata().frontend;
            syntax.reasoning = None;
            syntax.tools = Some(ToolParser {
                kind: case.kind,
                wrapping_newlines: 0,
                tag_separator: case.separator.into(),
            });
            let config = OutputConfig {
                syntax,
                control_tokens: controls.clone(),
                content: TaggedConfig {
                    structured: false,
                    start: case.start,
                    tools: vec![
                        json!({"type":"function","function":{"name":"weather","parameters":schema}}),
                    ],
                },
            };
            let envelope = OutputPipeline::grammar_envelope(&config).unwrap();
            let plan = ToolGrammarPlan {
                tools: vec![StrictToolSpec {
                    name: "weather".into(),
                    strict_schema: Some(&schema),
                }],
                syntax: Some((case.kind, 0, case.separator.into())),
                dsml_token: (case.kind == ToolParserKind::Dsml).then(|| {
                    controls
                        .iter()
                        .find(|(_, marker)| marker == "｜DSML｜")
                        .cloned()
                        .unwrap()
                }),
                allow_undeclared: false,
                require_call: true,
                single_call: true,
            };
            let encode = |wire: &str| {
                let mut tokens = Vec::new();
                let mut rest = wire;
                while let Some((at, id, marker)) = controls
                    .iter()
                    .filter_map(|(id, marker)| {
                        rest.find(marker).map(|at| (at, *id, marker.as_str()))
                    })
                    .min_by_key(|(at, _, _)| *at)
                {
                    tokens.extend(rest[..at].bytes().map(u32::from));
                    tokens.push(id);
                    rest = &rest[at + marker.len()..];
                }
                tokens.extend(rest.bytes().map(u32::from));
                tokens
            };

            let mut valid = compiler
                .compile_with_strict_tools(None, Some(&envelope), &plan)
                .unwrap();
            for (at, token) in encode(case.valid).into_iter().chain([256]).enumerate() {
                valid.advance(token).unwrap_or_else(|error| {
                    panic!(
                        "{} grammar rejected valid wire token {at}: {error}",
                        case.name
                    )
                });
            }

            let mut invalid = compiler
                .compile_with_strict_tools(None, Some(&envelope), &plan)
                .unwrap();
            let rejected = encode(case.invalid)
                .into_iter()
                .enumerate()
                .find_map(|(at, token)| invalid.advance(token).is_err().then_some(at));
            assert!(
                rejected.is_some(),
                "{} grammar accepted a schema-violating value through constrained generation",
                case.name
            );
        }
    }

    #[test]
    fn dsml_ordinary_marker_spelling_stays_inside_strict_string() {
        let marker = "｜DSML｜";
        let tokenizer = byte_tokenizer_with(&[marker]);
        let marker_id = tokenizer.token_to_id(marker).unwrap();
        let controls = vec![(marker_id, marker.to_owned())];
        let compiler = GrammarCompiler::new(&tokenizer, 320, &[256, 257], &controls).unwrap();
        let schema = json!({
            "type": "object",
            "properties": {"rationale": {"type": "string"}},
            "required": ["rationale"],
            "additionalProperties": false
        });
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = None;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::Dsml,
            wrapping_newlines: 0,
            tag_separator: " ".into(),
        });
        let config = OutputConfig {
            syntax,
            control_tokens: controls,
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![json!({"type":"function","function":{
                    "name":"note","strict":true,"parameters":schema
                }})],
            },
        };
        let envelope = OutputPipeline::grammar_envelope(&config).unwrap();
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "note".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::Dsml, 0, " ".into())),
            dsml_token: Some((marker_id, marker.into())),
            allow_undeclared: false,
            require_call: true,
            single_call: true,
        };
        let pieces = [
            (false, "<"),
            (true, marker),
            (false, " calls><"),
            (true, marker),
            (false, " invoke name=\"note\"><"),
            (true, marker),
            (
                false,
                " parameter name=\"rationale\" string=\"true\">before </",
            ),
            // The decoded spelling arrives without its control-token ID.
            (false, marker),
            (false, " parameter> after</"),
            (true, marker),
            (false, " parameter></"),
            (true, marker),
            (false, " invoke></"),
            (true, marker),
            (false, " calls>"),
        ];

        let mut grammar = compiler
            .compile_with_strict_tools(None, Some(&envelope), &plan)
            .unwrap();
        let mut parser = OutputPipeline::new(config);
        let mut events = Vec::new();
        for (special, text) in pieces {
            if special {
                grammar.advance(marker_id).unwrap();
                events.extend(parser.push(Input::Token(marker_id)).unwrap());
            } else {
                for byte in text.bytes() {
                    grammar.advance(u32::from(byte)).unwrap();
                }
            }
            events.extend(parser.push(Input::Text(text)).unwrap());
        }
        grammar.advance(256).unwrap();
        assert!(grammar.is_stopped());
        events.extend(parser.finish(Termination::Eos).unwrap());

        let arguments: String = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolArguments { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).unwrap(),
            json!({"rationale":"before </｜DSML｜ parameter> after"})
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::ToolBegin { name, .. } if name == "note"))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::ToolEnd { .. }))
                .count(),
            1
        );
    }

    #[test]
    #[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to DeepSeek-V4.1-Flash-1017"]
    fn ds41_checkpoint_strict_dsml_fixture_replays_complete_masks() {
        let snapshot = std::path::PathBuf::from(
            std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT")
                .expect("FLARE_DEEPSEEK_TEST_SNAPSHOT is required"),
        );
        let tokenizer = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json")).unwrap();
        assert_eq!(tokenizer.get_vocab_size(true), 129_280);
        assert_eq!(tokenizer.token_to_id("｜DSML｜"), Some(128_825));

        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 16,
                    "maxLength": 96,
                    "pattern": "^[A-Za-z0-9_ ./:-]+$"
                },
                "roots": {
                    "type": "array",
                    "prefixItems": [
                        {"const": "src"},
                        {"const": "tests"},
                        {"const": "docs"},
                        {"const": "examples"}
                    ],
                    "minItems": 4,
                    "maxItems": 4
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "case_sensitive": {"type": "boolean"},
                        "max_results": {
                            "type": "integer",
                            "minimum": 8,
                            "maximum": 32
                        },
                        "context_lines": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8
                        },
                        "include": {
                            "type": "array",
                            "prefixItems": [
                                {"const": "definitions"},
                                {"const": "references"},
                                {"const": "comments"},
                                {"const": "tests"}
                            ],
                            "minItems": 4,
                            "maxItems": 4
                        }
                    },
                    "required": [
                        "case_sensitive",
                        "max_results",
                        "context_lines",
                        "include"
                    ],
                    "additionalProperties": false
                },
                "rationale": {
                    "type": "string",
                    "minLength": 256,
                    "maxLength": 384
                }
            },
            "required": ["query", "roots", "options", "rationale"],
            "additionalProperties": false
        });
        let delimiters = [(128_825, "｜DSML｜".to_owned())];
        let compiler = GrammarCompiler::new(&tokenizer, 129_280, &[1], &delimiters).unwrap();
        let _validator =
            validate_strict_tool_schema(&compiler, "plan_local_code_search", &schema).unwrap();
        let envelope = OutputEnvelope {
            recipient: None,
            reasoning_end: None,
            tool: Some(("<｜DSML｜ calls>", "</｜DSML｜ calls>")),
            function: None,
            tool_ends_reasoning: false,
            tool_body_start: true,
            response: None,
            message_close: None,
        };
        let plan = ToolGrammarPlan {
            tools: vec![StrictToolSpec {
                name: "plan_local_code_search".into(),
                strict_schema: Some(&schema),
            }],
            syntax: Some((ToolParserKind::Dsml, 0, " ".into())),
            dsml_token: Some((128_825, "｜DSML｜".into())),
            allow_undeclared: false,
            require_call: true,
            single_call: true,
        };
        let wire = |query: &str, rationale: &str| {
            format!(
                "\n<｜DSML｜ invoke name=\"plan_local_code_search\">\n\
                 <｜DSML｜ parameter name=\"query\" string=\"true\">{query}</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"roots\" string=\"false\">[\"src\",\"tests\",\"docs\",\"examples\"]</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"options\" string=\"false\">{{\"case_sensitive\":false,\"max_results\":24,\"context_lines\":4,\"include\":[\"definitions\",\"references\",\"comments\",\"tests\"]}}</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"rationale\" string=\"true\">{rationale}</｜DSML｜ parameter>\n\
                 </｜DSML｜ invoke>\n\
                 </｜DSML｜ calls>"
            )
        };
        let rationale_prefix = "Inspect local CPU temperature definitions, references, comments, and tests across the four requested roots. Trace sensor reads, platform adapters, ownership, and error handling so the plan remains local and read-only. ";
        assert!(rationale_prefix.len() < 320);
        let rationale = format!(
            "{rationale_prefix}{}",
            "x".repeat(320 - rationale_prefix.len())
        );
        assert_eq!(rationale.chars().count(), 320);

        let replay = |query: &str, rationale: &str, accepted: bool| {
            let encoded = tokenizer.encode(wire(query, rationale), false).unwrap();
            assert_eq!(
                encoded
                    .get_ids()
                    .iter()
                    .filter(|&&token| token == 128_825)
                    .count(),
                11
            );
            let mut tokens = encoded.get_ids().to_vec();
            tokens.push(1);
            let mut grammar = compiler
                .compile_presented_strict_tools(None, Some(&envelope), &plan, None)
                .unwrap();
            let rejection = tokens.into_iter().enumerate().find_map(|(at, token)| {
                grammar.advance(token).err().map(|error| (at, token, error))
            });
            if accepted {
                assert!(
                    rejection.is_none(),
                    "valid fixture rejected at {rejection:?}"
                );
                assert!(grammar.is_stopped());
            } else {
                assert!(rejection.is_some(), "invalid fixture was accepted");
            }
        };

        replay("local CPU temperature source search", &rationale, true);
        replay("short", &rationale, false);
        replay(&"a".repeat(97), &rationale, false);
        replay("local CPU temperature?", &rationale, false);
        replay(
            "local CPU temperature source search",
            &"r".repeat(255),
            false,
        );
        replay(
            "local CPU temperature source search",
            &"r".repeat(385),
            false,
        );
    }
}
