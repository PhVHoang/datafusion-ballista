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

//! This module contains a dedicated thread pool for running "I/O
//! intensive" workloads such as shuffle file reads/writes and
//! Arrow Flight file serving.
//!
//! The IoExecutor is separate from the DedicatedExecutor (CPU-bound tasks)
//! to prevent I/O operations from blocking CPU-intensive query execution.

use log::{debug, warn};
use parking_lot::Mutex;
use std::{pin::Pin, sync::Arc};
use tokio::sync::oneshot::Receiver;

use futures::Future;

// The type of thing that the I/O executor runs
type Task = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Runs futures on a separate tokio runtime optimized for I/O-bound operations
/// like shuffle file reads/writes and Arrow Flight file serving.
///
/// Unlike DedicatedExecutor (CPU-bound), IoExecutor:
/// - Runs at normal thread priority (not low priority)
/// - Can oversubscribe threads (I/O threads block frequently)
/// - Optimized for blocking file operations
#[derive(Clone)]
pub struct IoExecutor {
    state: Arc<Mutex<State>>,
}

/// Internal state for the I/O executor
struct State {
    /// The number of threads in this pool
    num_threads: usize,

    /// The name of the threads for this executor
    thread_name: String,

    /// Channel for requests -- the I/O executor takes requests
    /// from here and runs them.
    requests: Option<std::sync::mpsc::Sender<Task>>,

    /// The thread that is doing the work
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for IoExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();

        let mut d = f.debug_struct("IoExecutor");

        d.field("num_threads", &state.num_threads)
            .field("thread_name", &state.thread_name);

        if state.requests.is_some() {
            d.field("requests", &"Some(...)")
        } else {
            d.field("requests", &"None")
        };

        if state.thread.is_some() {
            d.field("thread", &"Some(...)")
        } else {
            d.field("thread", &"None")
        };

        d.finish()
    }
}

impl IoExecutor {
    /// Creates a new `IoExecutor` with a dedicated tokio runtime
    /// optimized for I/O-bound operations.
    ///
    /// Unlike DedicatedExecutor:
    /// - Threads run at normal priority (I/O should not be deprioritized)
    /// - Designed for operations that frequently block on disk/network I/O
    ///
    /// # Arguments
    /// * `thread_name` - Base name for worker threads (will be suffixed with index)
    /// * `num_threads` - Number of worker threads. Recommended: 2-4x CPU cores
    ///                   since I/O threads spend most time blocked
    pub fn new(thread_name: impl Into<String>, num_threads: usize) -> Self {
        let thread_name = thread_name.into();
        let name_copy = thread_name.to_string();

        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn dedicated thread with tokio runtme for I/O operations
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name(&name_copy)
                .worker_threads(num_threads)
                // NOTE: No priority adjustment - I/O should run at normal priority
                .build()
                .expect("Creating tokio runtime for I/O executor");

            // By entering the context, all calls to `tokio::spawn` go
            // to this executor
            let _guard = runtime.enter();

            while let Ok(request) = rx.recv() {
                tokio::task::spawn(request);
            }
        });

        let state = State {
            num_threads,
            thread_name,
            requests: Some(tx),
            thread: Some(thread),
        };

        Self {
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Runs the specified Future (and any tasks it spawns) on the `IoExecutor`.
    ///
    /// Return a `Receiver` that will contain the result of the future when complete.
    /// Use `.await` on the receiver to get the result.
    ///
    /// # Example
    /// ```ignore
    /// let io_exec = IoExecutor::new("my-io", 4);
    /// let result = io_exec.spawn(async {
    ///     // Do some I/O operation
    ///     tokio::fs::read_to_string("file.txt").await
    /// }).await.unwrap();
    /// ```
    pub fn spawn<T>(&self, task: T) -> Receiver<T::Output>
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Create a job to spawn
        let job = Box::pin(async move {
            let task_output = task.await;
            if tx.send(task_output).is_err() {
                debug!("I/O task output ignored: receiver dropped");
            }
        });

        let mut state = self.state.lock();

        if let Some(requests) = &mut state.requests {
            requests.send(job).ok();
        } else {
            warn!("Tried to schedule I/O task on an executor that was shutdown");
        }

        rx
    }

    /// Signals shutdown of this executor and any clones.
    ///
    /// After calling shutdown, no new tasks will be accepted.
    /// Already-running tasks will continue to completion.
    pub fn shutdown(&self) {
        // Hang up the channel which will cause the dedicated thread to quit
        let mut state = self.state.lock();
        // Remaining jobs will still run to completion
        state.requests = None;
    }

    /// Stops all subsequent task executions, and waits for the worker
    /// thread to complete.
    ///
    /// Note this will shutdown all clones of this `IoExecutor` as well.
    /// Only the first call to `join` will actually wait for the executing
    /// thread to complete. All other calls will complete immediately.
    pub fn join(&self) {
        self.shutdown();

        // Take the thread out while mutex is held
        let thread = {
            let mut state = self.state.lock();
            state.thread.take()
        };

        // Wait for completion while not holding the mutex to avoid deadlocks
        if let Some(thread) = thread {
            thread.join().ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[tokio::test]
    async fn test_basic_io_task() {
        let exec = IoExecutor::new("test-io", 1);
        let result = exec.spawn(async { 42 }).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_io_executor_clone() {
        let barrier = Arc::new(Barrier::new(2));
        let exec = IoExecutor::new("test-io", 1);
        let task = exec.clone().spawn(do_work(42, Arc::clone(&barrier)));
        barrier.wait();
        assert_eq!(task.await.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_multi_task_concurrent() {
        let barrier = Arc::new(Barrier::new(3));

        // Make an executor with two threads
        let exec = IoExecutor::new("test-io", 2);
        let task1 = exec.spawn(do_work(11, Arc::clone(&barrier)));
        let task2 = exec.spawn(do_work(42, Arc::clone(&barrier)));

        // Block main thread until completion of other two tasks
        barrier.wait();

        // Should be able to get the results
        assert_eq!(task1.await.unwrap(), 11);
        assert_eq!(task2.await.unwrap(), 42);

        exec.join();
    }

    #[tokio::test]
    async fn test_tokio_spawn_on_io_executor() {
        let exec = IoExecutor::new("test-io", 2);

        // Spawn a task that spawns other tasks
        let result = exec
            .spawn(async move {
                // Spawn separate task
                let t1 = tokio::task::spawn(async {
                    assert_eq!(std::thread::current().name(), Some("test-io"));
                    25usize
                });
                t1.await.unwrap()
            })
            .await
            .unwrap();

        assert_eq!(result, 25);
    }

    #[tokio::test]
    async fn test_panic_on_io_executor() {
        let exec = IoExecutor::new("test-io", 1);
        let task = exec.spawn(async move {
            panic!("Panic in I/O task");
        });

        // Should not be able to get the result
        task.await.unwrap_err();
    }

    #[tokio::test]
    async fn test_shutdown_while_task_running() {
        let barrier_task_completed = Arc::new(Barrier::new(2));
        let barrier_task_running = Arc::new(Barrier::new(2));

        let exec = IoExecutor::new("test-io", 1);
        let task = exec.spawn(signal_running_do_work(
            42,
            Arc::clone(&barrier_task_running),
            Arc::clone(&barrier_task_completed),
        ));

        barrier_task_running.wait();
        exec.shutdown();

        // Block main thread until completion of the outstanding task
        barrier_task_completed.wait();

        // Task should complete successfully even after shutdown
        assert_eq!(task.await.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_submit_task_after_shutdown() {
        let exec = IoExecutor::new("test-io", 1);

        // Simulate trying to submit tasks once executor has shutdown
        exec.shutdown();
        let task = exec.spawn(async { 11 });

        // Task should complete, but return an error
        task.await.unwrap_err();
    }

    #[tokio::test]
    async fn test_submit_task_after_clone_shutdown() {
        let exec = IoExecutor::new("test-io", 1);

        // Shutdown the clone (but not the exec directly)
        exec.clone().join();

        // Simulate trying to submit tasks once executor has shutdown
        let task = exec.spawn(async { 11 });

        // Task should complete, but return an error
        task.await.unwrap_err();
    }

    #[tokio::test]
    async fn test_join() {
        let exec = IoExecutor::new("test-io", 1);
        // Test it doesn't hang
        exec.join()
    }

    #[tokio::test]
    #[allow(clippy::redundant_clone)]
    async fn test_clone_join() {
        let exec = IoExecutor::new("test-io", 1);
        // Test multiple joins don't hang
        exec.clone().join();
        exec.clone().join();
        exec.join();
    }

    /// Wait for the barrier and then return `result`
    async fn do_work(result: usize, barrier: Arc<Barrier>) -> usize {
        barrier.wait();
        result
    }

    /// Signals when task starts running, waits on barrier and then returns result
    async fn signal_running_do_work(
        result: usize,
        barrier_task_running: Arc<Barrier>,
        barrier_task_finished: Arc<Barrier>,
    ) -> usize {
        barrier_task_running.wait();
        barrier_task_finished.wait();
        result
    }
}
