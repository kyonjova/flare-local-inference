/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! ERNIE 5.5 XML tool-call dialect (`<ernie:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></ernie:tool_call>`).

use crate::Result;
use serde_json::{Map, Value};

const INVOKE: &str = "<invoke name=";
const INVOKE_END: &str = "</invoke>";
const PARAM: &str = "<parameter name=";
const PARAM_END: &str = "</parameter>";

fn coerce_parameter(raw: &str, schema: Option<&Value>) -> Value {
    let trimmed = raw.trim();
    if matches!(trimmed.to_ascii_lowercase().as_str(), "null" | "none" | "nil") {
        return Value::Null;
    }
    let mut types = Vec::new();
    super::chatml::parameter_types(schema, &mut types);
    if types.contains(&"boolean") {
        match trimmed.to_ascii_lowercase().as_str() {
            "yes" | "on" | "true" => return Value::Bool(true),
            "no" | "off" | "false" => return Value::Bool(false),
            _ => {}
        }
    }
    super::chatml::scalar_parameter_value(raw, schema)
}

pub struct ErnieXml;
impl super::tagged::ToolDialect for ErnieXml {
    const NAME: &'static str = "ERNIE XML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<ernie:tool_call>";
    const CLOSE: &'static str = "</ernie:tool_call>";
    const FUNCTION: Option<(&'static str, &'static str)> = Some((INVOKE, INVOKE_END));

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let mut rest = body.trim();
        let mut calls = Vec::new();
        while !rest.is_empty() {
            let invoke = rest
                .strip_prefix(INVOKE)
                .ok_or("ERNIE XML tool call requires an invoke tag")?;
            let (name, tail) = invoke
                .split_once('>')
                .ok_or("unterminated ERNIE XML function name")?;
            let name = name.trim().trim_matches(['\'', '"']);
            if name.is_empty() {
                return Err("tool function name is empty".into());
            }
            let properties = tools
                .iter()
                .find(|t| t["function"]["name"].as_str() == Some(name))
                .and_then(|tool| tool["function"].get("parameters"))
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object);
            let (mut parameters, tail) = tail
                .split_once(INVOKE_END)
                .ok_or("unterminated ERNIE XML invocation")?;
            let mut fields = Map::new();
            loop {
                parameters = parameters.trim_start();
                if parameters.is_empty() {
                    break;
                }
                let Some(after) = parameters.strip_prefix(PARAM) else {
                    break;
                };
                let Some((param_name, after_name)) = after.split_once('>') else {
                    break;
                };
                let param_name = param_name.trim().trim_matches(['\'', '"']);
                let Some((raw_value, next)) = after_name.split_once(PARAM_END) else {
                    break;
                };
                let schema = properties.and_then(|p| p.get(param_name));
                fields.insert(param_name.to_owned(), coerce_parameter(raw_value, schema));
                parameters = next;
            }
            calls.push((name.to_owned(), Value::Object(fields).to_string()));
            rest = tail.trim_start();
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tagged::ToolDialect;
    use serde_json::json;

    #[test]
    fn parses_parallel_invocations_and_coerces_types() {
        let body = r#"
<invoke name="get_weather"><parameter name="city">Paris</parameter><parameter name="days">3</parameter><parameter name="metric">yes</parameter><parameter name="note">null</parameter></invoke>
<invoke name="get_time"><parameter name="tz">CET</parameter></invoke>
"#;
        let tools = [json!({
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "metric": {"type": "boolean"},
                        "note": {"type": ["string", "null"]}
                    }
                }
            }
        })];
        let calls = ErnieXml::parse(body, &tools).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({"city": "Paris", "days": 3, "metric": true, "note": null})
        );
        assert_eq!(calls[1], ("get_time".into(), "{\"tz\":\"CET\"}".into()));
    }
}
