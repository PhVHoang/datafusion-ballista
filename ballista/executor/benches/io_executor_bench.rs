// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmarks comparing IoExecutor vs spawn_blocking for shuffle I/O operations

use ballista_executor::io_executor::IoExecutor;
use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use datafusion::arrow::array::{Int32Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::ipc::writer::FileWriter;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Create test shuffle files of varying sizes
fn create_test_shuffle_file(
    dir: &TempDir,
    num_batches: usize,
    rows_per_batch: usize,
) -> String {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let path = dir
        .path()
        .join(format!("shuffle_{}_{}.arrow", num_batches, rows_per_batch));
    let file = File::create(&path).unwrap();
    let mut writer = FileWriter::try_new(BufWriter::new(file), &schema).unwrap();

    for _ in 0..num_batches {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..rows_per_batch as i32)),
                Arc::new(Int32Array::from_iter_values(0..rows_per_batch as i32)),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }

    writer.finish().unwrap();
    path.to_str().unwrap().to_string()
}

/// Benchmark reading a shuffle file using spawn_blocking (baseline)
fn bench_spawn_blocking_read(runtime: &Runtime, path: &str) -> usize {
    runtime.block_on(async {
        let path = path.to_string();
        let handle = tokio::task::spawn_blocking(move || {
            let file = File::open(path).unwrap();
            let reader = datafusion::arrow::ipc::reader::FileReader::try_new(
                BufReader::new(file),
                None,
            )
            .unwrap();

            let mut total_rows = 0;
            for batch_result in reader {
                let batch = batch_result.unwrap();
                total_rows += batch.num_rows();
            }
            total_rows
        });

        handle.await.unwrap()
    })
}

/// Benchmark reading a shuffle file using IoExecutor
fn bench_io_executor_read(runtime: &Runtime, io_exec: &IoExecutor, path: &str) -> usize {
    runtime.block_on(async {
        let path = path.to_string();
        let result = io_exec
            .spawn(async move {
                let file = File::open(path).unwrap();
                let reader = datafusion::arrow::ipc::reader::FileReader::try_new(
                    BufReader::new(file),
                    None,
                )
                .unwrap();

                let mut total_rows = 0;
                for batch_result in reader {
                    let batch = batch_result.unwrap();
                    total_rows += batch.num_rows();
                }
                total_rows
            })
            .await
            .unwrap();

        result
    })
}

/// Benchmark concurrent reads - SYNC version for criterion benchmarks
fn bench_concurrent_reads(
    runtime: &Runtime,
    io_exec: Option<&IoExecutor>,
    paths: &[String],
) -> usize {
    runtime.block_on(bench_concurrent_reads_async(io_exec, paths))
}

/// Benchmark concurrent reads - ASYNC version for use within async contexts
async fn bench_concurrent_reads_async(
    io_exec: Option<&IoExecutor>,
    paths: &[String],
) -> usize {
    let mut handles = vec![];

    for path in paths {
        let path = path.to_string();

        let handle = match io_exec {
            Some(exec) => {
                // Using IoExecutor
                let rx = exec.spawn(async move {
                    let file = File::open(path).unwrap();
                    let reader = datafusion::arrow::ipc::reader::FileReader::try_new(
                        BufReader::new(file),
                        None,
                    )
                    .unwrap();

                    let mut total = 0;
                    for batch_result in reader {
                        total += batch_result.unwrap().num_rows();
                    }
                    total
                });
                tokio::spawn(async move { rx.await.unwrap() })
            }
            None => {
                // Using spawn_blocking
                tokio::spawn(async move {
                    tokio::task::spawn_blocking(move || {
                        let file = File::open(path).unwrap();
                        let reader = datafusion::arrow::ipc::reader::FileReader::try_new(
                            BufReader::new(file),
                            None,
                        )
                        .unwrap();

                        let mut total = 0;
                        for batch_result in reader {
                            total += batch_result.unwrap().num_rows();
                        }
                        total
                    })
                    .await
                    .unwrap()
                })
            }
        };

        handles.push(handle);
    }

    let mut total_rows = 0;
    for handle in handles {
        total_rows += handle.await.unwrap();
    }
    total_rows
}

fn benchmark_single_file_read(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let runtime = Runtime::new().unwrap();
    let io_executor = IoExecutor::new("bench-io", 4);

    // Test different file sizes
    let sizes = vec![
        ("small", 10, 1_000),    // ~40KB
        ("medium", 100, 10_000), // ~4MB
        ("large", 1000, 10_000), // ~40MB
    ];

    for (name, num_batches, rows_per_batch) in &sizes {
        let path = create_test_shuffle_file(&temp_dir, *num_batches, *rows_per_batch);
        let total_rows = num_batches * rows_per_batch;

        let mut group = c.benchmark_group(format!("single_file_{}", name));
        group.throughput(Throughput::Elements(total_rows as u64));

        group.bench_function("spawn_blocking", |b| {
            b.iter(|| bench_spawn_blocking_read(black_box(&runtime), black_box(&path)))
        });

        group.bench_function("io_executor", |b| {
            b.iter(|| {
                bench_io_executor_read(
                    black_box(&runtime),
                    black_box(&io_executor),
                    black_box(&path),
                )
            })
        });

        group.finish();
    }
}

fn benchmark_concurrent_reads(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let runtime = Runtime::new().unwrap();

    // Create multiple shuffle files
    let num_files = 16;
    let paths: Vec<String> = (0..num_files)
        .map(|_| create_test_shuffle_file(&temp_dir, 100, 10_000))
        .collect();

    // Test with different thread counts
    for thread_count in [2, 4, 8, 16] {
        let io_executor = IoExecutor::new("bench-io", thread_count);

        let mut group = c.benchmark_group(format!("concurrent_{}files", num_files));

        group.bench_function(BenchmarkId::new("spawn_blocking", "default"), |b| {
            b.iter(|| {
                bench_concurrent_reads(black_box(&runtime), None, black_box(&paths))
            })
        });

        group.bench_function(BenchmarkId::new("io_executor", thread_count), |b| {
            b.iter(|| {
                bench_concurrent_reads(
                    black_box(&runtime),
                    Some(black_box(&io_executor)),
                    black_box(&paths),
                )
            })
        });

        group.finish();
    }
}

fn benchmark_mixed_workload(c: &mut Criterion) {
    // Simulates realistic scenario: CPU-bound tasks + I/O tasks running together
    let temp_dir = TempDir::new().unwrap();
    let runtime = Runtime::new().unwrap();
    let io_executor = IoExecutor::new("bench-io", 8);

    let paths: Vec<String> = (0..8)
        .map(|_| create_test_shuffle_file(&temp_dir, 100, 10_000))
        .collect();

    let mut group = c.benchmark_group("mixed_workload");

    // Simulate CPU work
    async fn cpu_work() {
        for _ in 0..1000 {
            tokio::task::yield_now().await;
        }
    }

    group.bench_function("spawn_blocking_with_cpu", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let cpu_handle = tokio::spawn(cpu_work());
                let io_total = bench_concurrent_reads_async(None, &paths).await;
                cpu_handle.await.unwrap();
                io_total
            })
        })
    });

    group.bench_function("io_executor_with_cpu", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let cpu_handle = tokio::spawn(cpu_work());
                let io_total =
                    bench_concurrent_reads_async(Some(&io_executor), &paths).await;
                cpu_handle.await.unwrap();
                io_total
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_file_read,
    benchmark_concurrent_reads,
    benchmark_mixed_workload
);
criterion_main!(benches);
