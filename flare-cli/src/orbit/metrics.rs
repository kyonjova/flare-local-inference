/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! `flare orbit metrics`: the METRICS replies of a cluster's nodes as JSON.
//!
//! Each sample is one line: the cluster, the controller's clock, and one row per node holding
//! either the node's reply (its GPU readings with their holders and every group with the latest
//! metrics document of rank zero's serving process) or the error that kept it from answering.
//! `--watch` keeps one session per node open through the dashboard's poller and prints a
//! sample every second.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::controller::Controller;
use super::dashboard::{Update, poll_node};
use super::store::Node;
use super::{Failure, frame};

/// How often `--watch` prints; the agents sample their GPUs and groups at the same period.
const PERIOD: Duration = Duration::from_secs(1);

/// One sample: every node's latest result in cluster order. A node that has not answered yet
/// is reported unreachable without an error. Each group carries the `model` that `models` holds
/// for its name, or null.
fn sample(
    cluster: &str,
    nodes: &[Node],
    latest: &[Option<Result<Value, String>>],
    models: &BTreeMap<String, String>,
) -> Value {
    let rows: Vec<Value> = nodes
        .iter()
        .zip(latest)
        .map(|(node, result)| match result {
            Some(Ok(reply)) => {
                let mut row = json!({"name": node.name, "reachable": true});
                if let (Some(row), Some(fields)) = (row.as_object_mut(), reply.as_object()) {
                    row.extend(fields.clone());
                }
                for group in row["groups"].as_array_mut().into_iter().flatten() {
                    let model = group["name"].as_str().and_then(|name| models.get(name));
                    group["model"] = model.cloned().into();
                }
                row
            }
            Some(Err(error)) => json!({"name": node.name, "reachable": false, "error": error}),
            None => json!({"name": node.name, "reachable": false}),
        })
        .collect();
    json!({"cluster": cluster, "at": super::now_millis(), "nodes": rows})
}

/// Write one sample line; false once standard output is closed.
fn print(sample: &Value) -> bool {
    let mut output = io::stdout().lock();
    writeln!(output, "{sample}")
        .and_then(|()| output.flush())
        .is_ok()
}

/// Print the cluster's metrics once, or once a second with `watch` until standard output closes
/// or the process is interrupted.
pub fn run(controller: &Controller, cluster: Option<&str>, watch: bool) -> Result<(), Failure> {
    let (cluster, nodes) = controller.cluster_nodes(cluster)?;
    let models = controller.saved_models();
    if !watch {
        let latest: Vec<_> = thread::scope(|scope| {
            let handles: Vec<_> = nodes
                .iter()
                .map(|node| {
                    scope.spawn(move || {
                        let reply = controller
                            .node_session(node)
                            .and_then(|(mut session, _)| {
                                session.request(frame::METRICS, &json!({}))
                            })
                            .and_then(|body| Ok(serde_json::from_slice::<Value>(&body)?));
                        Some(reply.map_err(|error| error.to_string()))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("metrics request panicked"))
                .collect()
        });
        print(&sample(&cluster, &nodes, &latest, &models));
        return Ok(());
    }
    let (sender, receiver) = mpsc::channel::<Update>();
    let stop = Arc::new(AtomicBool::new(false));
    let mut latest: Vec<Option<Result<Value, String>>> = nodes.iter().map(|_| None).collect();
    thread::scope(|scope| {
        for (index, node) in nodes.iter().enumerate() {
            let sender = sender.clone();
            let stop = stop.clone();
            scope.spawn(move || poll_node(controller, node, index, sender, &stop));
        }
        drop(sender);
        let mut next = Instant::now() + PERIOD;
        loop {
            match receiver.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Ok(update) => latest[update.node] = Some(update.result),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    next += PERIOD;
                    if !print(&sample(&cluster, &nodes, &latest, &models)) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str) -> Node {
        Node {
            name: name.into(),
            fingerprint: String::new(),
            public_key: String::new(),
            address: "10.0.0.1:29900".into(),
        }
    }

    #[test]
    fn a_sample_carries_each_nodes_reply_or_its_error() {
        let nodes = [node("a"), node("b"), node("c")];
        let reply = json!({"at": 7, "gpus": [], "groups": [
            {"name": "g", "metrics": {"active": 2}}, {"name": "unsaved", "metrics": null}]});
        let latest = [Some(Ok(reply)), Some(Err("refused".to_owned())), None];
        let models = BTreeMap::from([("g".to_owned(), "GLM".to_owned())]);
        let sample = sample("sparks", &nodes, &latest, &models);
        let keys: Vec<&String> = sample["nodes"][0].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["name", "reachable", "at", "gpus", "groups"]);
        assert_eq!(sample["nodes"][0]["groups"][0]["model"], "GLM");
        assert_eq!(sample["nodes"][0]["groups"][1]["model"], Value::Null);
        assert_eq!(sample["cluster"], "sparks");
        assert!(sample["at"].as_u64().unwrap() > 0);
        assert_eq!(sample["nodes"][0]["name"], "a");
        assert_eq!(sample["nodes"][0]["reachable"], true);
        assert_eq!(sample["nodes"][0]["at"], 7);
        assert_eq!(sample["nodes"][0]["groups"][0]["metrics"]["active"], 2);
        assert_eq!(
            sample["nodes"][1],
            json!({"name": "b", "reachable": false, "error": "refused"})
        );
        assert_eq!(sample["nodes"][2], json!({"name": "c", "reachable": false}));
    }
}
