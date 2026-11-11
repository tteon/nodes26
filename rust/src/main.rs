use neo4rs::{query, ConfigBuilder, Graph};
use serde_json::{json, Value};
use std::{
    env,
    io::{self, Read},
    time::Instant,
};
use tokio::task::JoinSet;
async fn branch(g: Graph, input: Value, db: String, cats: Vec<String>, epoch: Instant) -> Value {
    let start = epoch.elapsed().as_nanos() as u64;
    let mut tx = g.start_txn_on(db.clone()).await.expect("begin");
    let q = query(input["query"].as_str().unwrap())
        .param("cik", input["cik"].as_str().unwrap())
        .param("contexts", cats);
    let mut stream = tx.execute(q).await.expect("execute");
    let mut rows: Vec<String> = Vec::new();
    while let Some(row) = stream.next(&mut tx).await.expect("consume") {
        rows.push(row.get::<String>("payload").expect("payload"));
    }
    tx.commit().await.expect("commit");
    json!({"database":db,"start_ns":start,"end_ns":epoch.elapsed().as_nanos() as u64,"rows":rows})
}
async fn request(g: Graph, i: Value, epoch: Instant, id: usize) -> Value {
    let start = epoch.elapsed().as_nanos() as u64;
    let mode = i["mode"].as_str().unwrap();
    let all: Vec<String> = i["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let targets: Vec<(String, Vec<String>)> = if mode == "single" {
        vec![("neo4j".into(), all.clone())]
    } else {
        vec![
            ("ctxa".into(), vec![all[0].clone()]),
            ("ctxb".into(), vec![all[1].clone()]),
        ]
    };
    let mut branches = Vec::new();
    if mode == "parallel" {
        let mut jobs = JoinSet::new();
        for (db, cats) in targets {
            jobs.spawn(branch(g.clone(), i.clone(), db, cats, epoch));
        }
        while let Some(b) = jobs.join_next().await {
            branches.push(b.expect("branch task"));
        }
    } else {
        for (db, cats) in targets {
            branches.push(branch(g.clone(), i.clone(), db, cats, epoch).await);
        }
    }
    let mut rows: Vec<String> = branches
        .iter()
        .flat_map(|b| {
            b["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
        })
        .collect();
    rows.sort();
    json!({"id":id,"start_ns":start,"end_ns":epoch.elapsed().as_nanos() as u64,"branches":branches,"rows":rows})
}
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let i: Value = serde_json::from_str(&s).unwrap();
    let cfg = ConfigBuilder::default()
        .uri(env::var("DOZER_URI").unwrap())
        .user(env::var("DOZER_USER").unwrap())
        .password(env::var("DOZER_PASSWORD").unwrap())
        .fetch_size(128)
        .max_connections(8)
        .build()
        .unwrap();
    let g = Graph::connect(cfg).await.expect("connect");
    let epoch = Instant::now();
    for n in 0..8 {
        request(g.clone(), i.clone(), epoch, n).await;
    }
    let concurrency = i["concurrency"].as_u64().unwrap() as usize;
    let count = i["count"].as_u64().unwrap() as usize;
    let started = epoch.elapsed().as_nanos() as u64;
    let mut jobs = JoinSet::new();
    let mut rows = Vec::new();
    for id in 0..count {
        if jobs.len() >= concurrency {
            rows.push(jobs.join_next().await.unwrap().unwrap());
        }
        jobs.spawn(request(g.clone(), i.clone(), epoch, id));
    }
    while let Some(r) = jobs.join_next().await {
        rows.push(r.unwrap());
    }
    println!(
        "{}",
        json!({"samples":rows,"batch_start_ns":started,"batch_end_ns":epoch.elapsed().as_nanos() as u64})
    );
}
