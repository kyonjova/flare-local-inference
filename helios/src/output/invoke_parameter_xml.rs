/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! XML `<invoke name="...">` tool calls with `<parameter name="...">` arguments.

use crate::Result;
use serde_json::{Map, Value};

const INVOKE: &str = "<invoke name=";
const INVOKE_END: &str = "</invoke>";
const PARAMETER_END: &str = "</parameter>";

/// Parameter tags read by `InvokeParameterXml::parse`; `OutputPipeline::declared_markers`
/// lists them alongside `ToolDialect` delimiters.
pub(crate) const PARAMETER_MARKERS: [&str; 2] = ["<parameter name=", PARAMETER_END];

fn parse_named_open<'a>(text: &'a str, tag: &str) -> Option<(&'a str, &'a str)> {
    let rest = text.strip_prefix('<')?.strip_prefix(tag)?;
    let ws_len = rest.len() - rest.trim_start().len();
    if ws_len == 0 {
        return None;
    }
    let rest = rest[ws_len..].strip_prefix("name")?.trim_start().strip_prefix('=')?.trim_start();
    let (name, after_name) = if let Some(inner) = rest.strip_prefix('"') {
        let (name, tail) = inner.split_once('"')?;
        let tail = tail.trim_start().strip_prefix('>')?;
        (name, tail)
    } else if let Some(inner) = rest.strip_prefix('\'') {
        let (name, tail) = inner.split_once('\'')?;
        let tail = tail.trim_start().strip_prefix('>')?;
        (name, tail)
    } else {
        let (name, tail) = rest.split_once('>')?;
        let name = name.trim();
        if name.contains('<') || name.contains('"') || name.contains('\'') {
            return None;
        }
        (name, tail)
    };
    if name.is_empty() {
        return None;
    }
    Some((name, after_name))
}

fn possible_named_open(text: &str, tag: &str) -> bool {
    let Some(rest) = text.strip_prefix('<') else {
        return false;
    };
    if tag.starts_with(rest) {
        return true;
    }
    let Some(rest) = rest.strip_prefix(tag) else {
        return false;
    };
    let trimmed = rest.trim_start();
    if trimmed.len() == rest.len() {
        return false;
    }
    if trimmed.is_empty() || "name".starts_with(trimmed) {
        return true;
    }
    let Some(after_name) = trimmed.strip_prefix("name") else {
        return false;
    };
    let after_name = after_name.trim_start();
    if after_name.is_empty() {
        return true;
    }
    let Some(after_eq) = after_name.strip_prefix('=') else {
        return false;
    };
    let after_eq = after_eq.trim_start();
    if after_eq.is_empty() {
        return true;
    }
    if let Some(inner) = after_eq.strip_prefix('"') {
        if let Some((name, tail)) = inner.split_once('"') {
            if name.is_empty() {
                return false;
            }
            let tail = tail.trim_start();
            return tail.is_empty() || tail.starts_with('>');
        }
        return true;
    }
    if let Some(inner) = after_eq.strip_prefix('\'') {
        if let Some((name, tail)) = inner.split_once('\'') {
            if name.is_empty() {
                return false;
            }
            let tail = tail.trim_start();
            return tail.is_empty() || tail.starts_with('>');
        }
        return true;
    }
    if let Some((name, _)) = after_eq.split_once('>') {
        let name = name.trim();
        !name.is_empty() && !name.contains('<') && !name.contains('"') && !name.contains('\'')
    } else {
        !after_eq.contains('<') && !after_eq.contains('"') && !after_eq.contains('\'')
    }
}

fn lookup_pointer<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let path = reference.strip_prefix('#')?;
    if path.is_empty() {
        return Some(root);
    }
    let path = path.strip_prefix('/')?;
    let mut node = root;
    for raw_part in path.split('/') {
        let part = raw_part.replace("~1", "/").replace("~0", "~");
        match node {
            Value::Object(map) => node = map.get(&part)?,
            Value::Array(items) => {
                let index: usize = part.parse().ok()?;
                node = items.get(index)?;
            }
            _ => return None,
        }
    }
    Some(node)
}

fn resolve_refs(schema: &Value, root: &Value, seen: &mut Vec<String>) -> Value {
    let Some(map) = schema.as_object() else {
        return schema.clone();
    };
    if let Some(reference) = map.get("$ref").and_then(Value::as_str)
        && reference.starts_with('#')
        && !seen.iter().any(|item| item == reference)
        && let Some(target) = lookup_pointer(root, reference).and_then(Value::as_object)
    {
        let mut merged = target.clone();
        for (key, value) in map {
            if key != "$ref" {
                merged.insert(key.clone(), value.clone());
            }
        }
        seen.push(reference.to_owned());
        let resolved = resolve_refs(&Value::Object(merged), root, seen);
        seen.pop();
        return resolved;
    }
    let mut resolved = map.clone();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = map.get(keyword).and_then(Value::as_array) {
            let expanded = branches
                .iter()
                .map(|branch| resolve_refs(branch, root, seen))
                .collect();
            resolved.insert(keyword.to_owned(), Value::Array(expanded));
        }
    }
    Value::Object(resolved)
}

fn schema_properties(schema: &Value, root: &Value, seen: &mut Vec<String>) -> Map<String, Value> {
    let resolved = resolve_refs(schema, root, seen);
    if let Some(properties) = resolved.get("properties").and_then(Value::as_object) {
        return properties.clone();
    }
    let mut merged = Map::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = resolved.get(keyword).and_then(Value::as_array) {
            for branch in branches {
                for (key, value) in schema_properties(branch, root, seen) {
                    merged.entry(key).or_insert(value);
                }
            }
        }
    }
    merged
}

fn coerce_parameter_value(raw: &str, schema: Option<&Value>) -> Value {
    let lowered = raw.trim().to_ascii_lowercase();
    if matches!(lowered.as_str(), "null" | "none" | "nil") {
        return Value::Null;
    }
    let mut kinds = Vec::new();
    super::chatml::parameter_types(schema, &mut kinds);
    if kinds
        .iter()
        .any(|kind| kind.eq_ignore_ascii_case("boolean") || kind.eq_ignore_ascii_case("bool"))
    {
        match lowered.as_str() {
            "yes" | "on" => return Value::Bool(true),
            "no" | "off" => return Value::Bool(false),
            _ => {}
        }
    }
    super::chatml::scalar_parameter_value(raw, schema)
}

pub struct InvokeParameterXml;
impl super::tagged::ToolDialect for InvokeParameterXml {
    const NAME: &'static str = "invoke-parameter XML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<ernie:tool_call>";
    const CLOSE: &'static str = "</ernie:tool_call>";
    const FUNCTION: Option<(&'static str, &'static str)> = Some((INVOKE, INVOKE_END));

    fn starts_call(text: &str) -> Option<bool> {
        let text = text.strip_prefix(Self::OPEN).map_or(text, str::trim_start);
        if text.is_empty() {
            return None;
        }
        let Some((_, body)) = parse_named_open(text, "invoke") else {
            return if possible_named_open(text, "invoke") {
                None
            } else {
                Some(false)
            };
        };
        let body = body.trim_start();
        if body.is_empty() {
            return None;
        }
        if parse_named_open(body, "parameter").is_some() || body.starts_with(INVOKE_END) {
            Some(true)
        } else if possible_named_open(body, "parameter") || INVOKE_END.starts_with(body) {
            None
        } else {
            Some(false)
        }
    }

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let mut rest = body.trim();
        let mut calls = Vec::new();
        while !rest.is_empty() {
            let (name, mut tail) = parse_named_open(rest, "invoke")
                .ok_or("XML tool call requires a valid invoke tag")?;
            let root_schema = tools
                .iter()
                .filter_map(|tool| tool.get("function"))
                .find(|function| function.get("name").and_then(Value::as_str) == Some(name))
                .and_then(|function| function.get("parameters"));
            let properties = root_schema.map(|root| {
                let mut seen = Vec::new();
                schema_properties(root, root, &mut seen)
            });
            let mut arguments = Map::new();
            loop {
                tail = tail.trim_start();
                if let Some(after_invoke) = tail.strip_prefix(INVOKE_END) {
                    rest = after_invoke.trim_start();
                    break;
                }
                let (key, value_start) = parse_named_open(tail, "parameter")
                    .ok_or("XML invocation requires a parameter or closing invoke tag")?;
                if arguments.contains_key(key) {
                    return Err("duplicate XML tool parameter".into());
                }
                let (raw_value, after_param) = value_start
                    .split_once(PARAMETER_END)
                    .ok_or("unterminated XML tool parameter")?;
                let resolved_schema = root_schema.and_then(|root| {
                    let prop = properties.as_ref()?.get(key)?;
                    let mut seen = Vec::new();
                    Some(resolve_refs(prop, root, &mut seen))
                });
                let coerced = coerce_parameter_value(raw_value, resolved_schema.as_ref());
                arguments.insert(key.to_owned(), coerced);
                tail = after_param;
            }
            calls.push((name.to_owned(), Value::Object(arguments).to_string()));
        }
        if calls.is_empty() {
            return Err("empty XML tool call block".into());
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tagged::ToolDialect;
    use serde_json::json;

    fn convert_tools() -> Vec<Value> {
        vec![json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "$defs": {
                        "N": {"type": "integer"},
                        "Slot": {"type": "object"},
                        "Cycle": {"$ref": "#/$defs/Cycle"}
                    },
                    "properties": {
                        "s": {"type": "string"},
                        "b": {"type": "boolean"},
                        "n": {"$ref": "#/$defs/N"},
                        "slot": {"$ref": "#/$defs/Slot"},
                        "u": {"anyOf": [{"$ref": "#/$defs/N"}, {"type": "boolean"}]},
                        "cycle": {"$ref": "#/$defs/Cycle"}
                    }
                }
            }
        })]
    }

    #[test]
    fn parses_parallel_invocations_and_zero_arg_calls() {
        let tools = vec![
            json!({"function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}),
            json!({"function":{"name":"ping","parameters":{"type":"object","properties":{}}}}),
        ];
        let body = r#"
<invoke name="get_weather"><parameter name="city">Paris</parameter><parameter name="days">3</parameter></invoke>
<invoke name='ping'></invoke>
<invoke name=get_weather><parameter name=city>Rome</parameter></invoke>
"#;
        let calls = InvokeParameterXml::parse(body, &tools).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({"city": "Paris", "days": 3})
        );
        assert_eq!(calls[1], ("ping".into(), "{}".into()));
        assert_eq!(calls[2].0, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[2].1).unwrap(),
            json!({"city": "Rome"})
        );
    }

    #[test]
    fn resolves_local_refs_combinators_null_words_and_boolean_words() {
        let tools = convert_tools();
        let cases = [
            (
                r#"<invoke name="f"><parameter name="s">null</parameter><parameter name="b">None</parameter><parameter name="n">nil</parameter></invoke>"#,
                json!({"s": null, "b": null, "n": null}),
            ),
            (
                r#"<invoke name="f"><parameter name="b">yes</parameter></invoke>"#,
                json!({"b": true}),
            ),
            (
                r#"<invoke name="f"><parameter name="b">Off</parameter></invoke>"#,
                json!({"b": false}),
            ),
            (
                r#"<invoke name="f"><parameter name="b">maybe</parameter></invoke>"#,
                json!({"b": "maybe"}),
            ),
            (
                r#"<invoke name="f"><parameter name="n">5</parameter><parameter name="slot">{"start": "9:00"}</parameter></invoke>"#,
                json!({"n": 5, "slot": {"start": "9:00"}}),
            ),
            (
                r#"<invoke name="f"><parameter name="u">on</parameter><parameter name="s">yes</parameter></invoke>"#,
                json!({"u": true, "s": "yes"}),
            ),
            (
                r#"<invoke name="f"><parameter name="cycle">raw</parameter></invoke>"#,
                json!({"cycle": "raw"}),
            ),
        ];
        for (body, expected) in cases {
            let calls = InvokeParameterXml::parse(body, &tools).unwrap();
            let parsed: Value = serde_json::from_str(&calls[0].1).unwrap();
            assert_eq!(parsed, expected, "{body}");
        }
    }

    #[test]
    fn merges_top_level_anyof_properties() {
        let tools = vec![json!({
            "function": {
                "name": "union",
                "parameters": {
                    "anyOf": [
                        {"type": "object", "properties": {"a": {"type": "integer"}}},
                        {"type": "object", "properties": {"b": {"type": "boolean"}}}
                    ]
                }
            }
        })];
        let body = r#"<invoke name="union"><parameter name="a">7</parameter><parameter name="b">no</parameter></invoke>"#;
        let calls = InvokeParameterXml::parse(body, &tools).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({"a": 7, "b": false})
        );
    }

    #[test]
    fn rejects_malformed_empty_and_duplicate_invocations() {
        let tools = convert_tools();
        for body in [
            "",
            "   ",
            r#"<invoke name=""></invoke>"#,
            r#"<invoke name="f"><parameter name="">1</parameter></invoke>"#,
            r#"<invoke name="f"><parameter name="s">a</parameter><parameter name="s">b</parameter></invoke>"#,
            r#"<invoke name="f"><parameter name="s">unclosed</invoke>"#,
            r#"<invoke name="f"><parameter name="s">ok</parameter>"#,
            r#"<invoke name="f">stray</invoke>"#,
            r#"not an invoke"#,
        ] {
            assert!(
                InvokeParameterXml::parse(body, &tools).is_err(),
                "expected error for {body:?}"
            );
        }
    }

    #[test]
    fn starts_call_distinguishes_calls_from_prose() {
        for (text, expected) in [
            ("<ernie:tool_call>", None),
            ("<ernie:tool_call>\n<inv", None),
            ("<ernie:tool_call>\n<invoke name=\"f\">", None),
            ("<ernie:tool_call>\n<invoke name=\"f\"><param", None),
            ("<ernie:tool_call>\n<invoke name=\"f\"></inv", None),
            ("<ernie:tool_call>\n<invoke name=\"f\"></invoke>", Some(true)),
            (
                "<ernie:tool_call>\n<invoke name=\"f\"><parameter name=\"x\">",
                Some(true),
            ),
            ("<ernie:tool_call> in prose", Some(false)),
            ("<invoke name=\"\">", Some(false)),
            ("<invoke name=\"f\"> prose", Some(false)),
        ] {
            assert_eq!(InvokeParameterXml::starts_call(text), expected, "{text:?}");
        }
    }
}
