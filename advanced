# MPI Advanced Patterns and Edge Cases

This document covers implementation details, edge cases, and advanced techniques for protocol-correct, deadlock-free MPI applications.

## Table of Contents
1. [Protocol Correctness Edge Cases](#protocol-correctness-edge-cases)
2. [Deadlock Detection Strategies](#deadlock-detection-strategies)
3. [Advanced Congestion Patterns](#advanced-congestion-patterns)
4. [Error Recovery](#error-recovery)
5. [Performance Anti-Patterns](#performance-anti-patterns)
6. [Production Debugging Tools](#production-debugging-tools)

---

## Protocol Correctness Edge Cases

### Case 1: Unexpected Message Order

**Problem:** Process A sends message X with tag 1, then Y with tag 2. Process B does `Recv(..., MPI_ANY_TAG)` twice. Message order is not guaranteed to match send order.

```cpp
// Process A
MPI_Send(x, 1, MPI_INT, 1, 1, comm);
MPI_Send(y, 1, MPI_INT, 1, 2, comm);

// Process B
int buf1, buf2;
MPI_Recv(&buf1, 1, MPI_INT, 0, MPI_ANY_TAG, comm, &status);  // Gets Y!
MPI_Recv(&buf2, 1, MPI_INT, 0, MPI_ANY_TAG, comm, &status);  // Gets X
```

**Solution:** Always specify tags or use separate channels for ordered sequences.

```cpp
// SAFE: Process B expects both tags
MPI_Recv(&buf1, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
MPI_Recv(&buf2, 1, MPI_INT, 0, 2, comm, MPI_STATUS_IGNORE);
```

### Case 2: MPI_Send Blocking Behavior

`MPI_Send` is **not necessarily non-blocking**. It may block if:
- System buffer space is exhausted (typically ~64 KB)
- Receiver is not ready and eager protocol not available

```cpp
// ❌ May deadlock on large messages
for (int i = 0; i < size; ++i) {
    if (rank == i) {
        for (int j = 0; j < size; ++j) {
            if (i != j) {
                MPI_Send(data, LARGE_SIZE, MPI_INT, j, 0, comm);  // Blocks if buffer full
            }
        }
    }
}
```

**Safe alternative:**

```cpp
// ✅ Use Isend for guaranteed non-blocking
std::vector<MPI_Request> reqs;
for (int j = 0; j < size; ++j) {
    if (rank != j) {
        MPI_Request req;
        MPI_Isend(data, LARGE_SIZE, MPI_INT, j, 0, comm, &req);
        reqs.push_back(req);
    }
}
MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
```

### Case 3: Untyped Data Corruption

MPI doesn't enforce type consistency across message boundaries.

```cpp
// Process A sends
int intdata[3] = {1, 2, 3};
MPI_Send(intdata, 3, MPI_INT, 1, 0, comm);

// Process B receives as float
float floatdata[3];
MPI_Recv(floatdata, 3, MPI_FLOAT, 0, 0, comm, MPI_STATUS_IGNORE);  // Data corrupted!
```

**Solution:** Use strongly typed messages or `MPI_Pack`.

```cpp
// Option 1: Explicit type on both sides
int intdata[3] = {1, 2, 3};
MPI_Send(intdata, 3, MPI_INT, 1, 0, comm);  // Send as INT
// On receiver
int recvdata[3];
MPI_Recv(recvdata, 3, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);  // Recv as INT

// Option 2: Use MPI_Pack for heterogeneous data
int pack_position = 0;
char pack_buffer[256];
MPI_Pack(&x, 1, MPI_INT, pack_buffer, 256, &pack_position, comm);
MPI_Pack(&y, 1, MPI_DOUBLE, pack_buffer, 256, &pack_position, comm);
MPI_Send(pack_buffer, pack_position, MPI_PACKED, 1, 0, comm);

// Receiver unpack
pack_position = 0;
MPI_Unpack(pack_buffer, pack_position, 256, &x, 1, MPI_INT, comm);
MPI_Unpack(pack_buffer, pack_position, 256, &y, 1, MPI_DOUBLE, comm);
```

---

## Deadlock Detection Strategies

### Strategy 1: Timeout-Based Detection

```cpp
#include <mpi.h>
#include <signal.h>

volatile int timeout_occurred = 0;

void timeout_handler(int sig) {
    timeout_occurred = 1;
}

int safe_recv_with_timeout(void* buf, int count, MPI_Datatype dtype, int src, 
                           int tag, MPI_Comm comm, int timeout_secs) {
    signal(SIGALRM, timeout_handler);
    alarm(timeout_secs);
    
    MPI_Recv(buf, count, dtype, src, tag, comm, MPI_STATUS_IGNORE);
    
    alarm(0);  // Cancel alarm
    
    if (timeout_occurred) {
        fprintf(stderr, "TIMEOUT: Recv from rank %d tag %d did not complete\n", src, tag);
        return MPI_ERR_OTHER;
    }
    return MPI_SUCCESS;
}
```

### Strategy 2: Non-Blocking with Progress Polling

```cpp
int safe_recv_with_polling(void* buf, int count, MPI_Datatype dtype, int src, 
                          int tag, MPI_Comm comm, int max_polls) {
    MPI_Request req;
    MPI_Irecv(buf, count, dtype, src, tag, comm, &req);
    
    for (int poll = 0; poll < max_polls; ++poll) {
        int flag;
        MPI_Status status;
        
        // Check if message arrived
        MPI_Test(&req, &flag, &status);
        if (flag) return MPI_SUCCESS;
        
        // Try to make progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
        if (poll % 100 == 0) {
            fprintf(stderr, "[Poll %d/%d] Waiting for message from %d tag %d\n", 
                   poll, max_polls, src, tag);
        }
    }
    
    fprintf(stderr, "ERROR: Recv did not complete after %d polls\n", max_polls);
    MPI_Cancel(&req);
    return MPI_ERR_PENDING;
}
```

### Strategy 3: Graph-Based Deadlock Detection

For complex communication patterns, verify acyclicity:

```cpp
// Build dependency graph
// Edge: Process A → Process B if A sends to B
// Deadlock occurs ⟺ graph has cycle

bool has_cycle(int** adj_matrix, int size) {
    // DFS to detect cycles
    std::vector<int> color(size, 0);  // 0: white, 1: gray, 2: black
    
    for (int start = 0; start < size; ++start) {
        if (color[start] == 0) {
            std::stack<int> st;
            st.push(start);
            color[start] = 1;
            
            while (!st.empty()) {
                int u = st.top();
                bool found_gray = false;
                
                for (int v = 0; v < size; ++v) {
                    if (adj_matrix[u][v] && color[v] == 1) {
                        return true;  // Back edge found → cycle
                    }
                    if (adj_matrix[u][v] && color[v] == 0) {
                        color[v] = 1;
                        st.push(v);
                        found_gray = true;
                        break;
                    }
                }
                
                if (!found_gray) {
                    color[u] = 2;
                    st.pop();
                }
            }
        }
    }
    return false;
}
```

---

## Advanced Congestion Patterns

### Pattern 1: Hierarchical Reduction (Tree-Based)

```cpp
void tree_reduction(int rank, int size, int* data, int count, 
                   MPI_Op op, MPI_Comm comm) {
    int level = 0;
    int stride = 1;
    
    while (stride < size) {
        int parent = rank - (rank % (stride * 2));  // Parent in next level
        int left_child = rank;
        int right_child = rank + stride;
        
        if (right_child < size && (rank % (stride * 2)) == 0) {
            // Parent: receive from right child
            MPI_Recv(data, count, MPI_INT, right_child, level, comm, MPI_STATUS_IGNORE);
            // Apply operation locally
            for (int i = 0; i < count; ++i) {
                if (op == MPI_SUM) data[i] += right_child_data[i];
            }
        }
        
        if ((rank % (stride * 2)) != 0) {
            // Non-root: send to parent
            MPI_Send(data, count, MPI_INT, parent, level, comm);
        }
        
        MPI_Barrier(comm);  // Synchronize level
        stride *= 2;
        level++;
    }
}
```

**Complexity:**
- Message count: O(n log n) instead of O(n²) for all-to-all
- Network hops: O(log n)
- Congestion: Distributed across tree levels

### Pattern 2: 2D Mesh Reduction

For grid-based applications, use 2D communication:

```cpp
void mesh_reduction(int rank, int row_rank, int col_rank, int* data, int count,
                   int row_size, int col_size, MPI_Comm row_comm, MPI_Comm col_comm) {
    // Step 1: Reduce within rows
    MPI_Allreduce(data, data, count, MPI_INT, MPI_SUM, row_comm);
    
    // Step 2: Reduce within columns (only rank 0 of each column participates)
    if (col_rank == 0) {
        MPI_Allreduce(data, data, count, MPI_INT, MPI_SUM, col_comm);
    }
    
    // Step 3: Broadcast result to all
    MPI_Bcast(data, count, MPI_INT, 0, row_comm);
    MPI_Bcast(data, count, MPI_INT, 0, col_comm);
}
```

**Congestion reduction:**
- Row phase: Each row independent → parallel
- Column phase: Only (1/row_size) of messages
- Total: O(log n) latency, reduced bandwidth per link

### Pattern 3: Overlapped Computation-Communication

```cpp
void overlapped_stencil(int rank, int* local_data, int local_size,
                       int* left_ghost, int* right_ghost,
                       MPI_Comm comm) {
    int left_rank = rank - 1;
    int right_rank = rank + 1;
    
    // Post non-blocking sends (halo exchange)
    MPI_Request send_reqs[2], recv_reqs[2];
    MPI_Isend(&local_data[1], 1, MPI_INT, left_rank, 0, comm, &send_reqs[0]);
    MPI_Isend(&local_data[local_size - 2], 1, MPI_INT, right_rank, 0, comm, &send_reqs[1]);
    MPI_Irecv(left_ghost, 1, MPI_INT, left_rank, 0, comm, &recv_reqs[0]);
    MPI_Irecv(right_ghost, 1, MPI_INT, right_rank, 0, comm, &recv_reqs[1]);
    
    // Compute interior elements (don't need ghosts)
    for (int i = 2; i < local_size - 2; ++i) {
        local_data[i] = (local_data[i-1] + local_data[i+1]) / 2.0;
    }
    
    // Wait for ghost exchange
    MPI_Waitall(2, recv_reqs, MPI_STATUSES_IGNORE);
    
    // Compute boundary elements
    local_data[1] = (left_ghost[0] + local_data[2]) / 2.0;
    local_data[local_size - 2] = (local_data[local_size - 3] + right_ghost[0]) / 2.0;
    
    // Ensure sends complete
    MPI_Waitall(2, send_reqs, MPI_STATUSES_IGNORE);
}
```

**Benefit:** While waiting for ghost cells, interior is computed in parallel with communication.

---

## Error Recovery

### Communicator Abort

```cpp
int safe_operation(void* buf, int count, MPI_Datatype dtype, int dest,
                  int tag, MPI_Comm comm) {
    int ret = MPI_Send(buf, count, dtype, dest, tag, comm);
    
    if (ret != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_len;
        MPI_Error_string(ret, error_string, &error_len);
        fprintf(stderr, "FATAL: Send failed: %s\n", error_string);
        MPI_Abort(comm, ret);
    }
    return ret;
}
```

### Graceful Shutdown with Barrier

```cpp
void graceful_shutdown(MPI_Comm comm, int error_code) {
    // Notify all processes of shutdown
    MPI_Bcast(&error_code, 1, MPI_INT, 0, comm);
    
    // Synchronize before exit
    MPI_Barrier(comm);
    
    if (error_code != MPI_SUCCESS) {
        MPI_Finalize();
        exit(1);
    }
}
```

---

## Performance Anti-Patterns

### Anti-Pattern 1: Fine-Grained Communication Loop

```cpp
// ❌ 1000+ small messages = latency-bound, congested
for (int i = 0; i < 1000; ++i) {
    int value = some_computation();
    MPI_Send(&value, 1, MPI_INT, neighbor, i, comm);
    MPI_Recv(&neighbor_value, 1, MPI_INT, neighbor, i, comm, MPI_STATUS_IGNORE);
}
```

**Fix: Batch communications**

```cpp
// ✅ Single large message = bandwidth-bound, efficient
std::vector<int> send_buf(1000), recv_buf(1000);
for (int i = 0; i < 1000; ++i) {
    send_buf[i] = some_computation();
}
MPI_Sendrecv(send_buf.data(), 1000, MPI_INT, neighbor, 0,
            recv_buf.data(), 1000, MPI_INT, neighbor, 0,
            comm, MPI_STATUS_IGNORE);
```

### Anti-Pattern 2: Synchronous Waits

```cpp
// ❌ Process waits for every operation sequentially
for (int i = 0; i < 100; ++i) {
    MPI_Isend(&data[i], 1, MPI_INT, neighbor, i, comm, &req[i]);
    MPI_Wait(&req[i], MPI_STATUS_IGNORE);  // Wait immediately — defeats non-blocking
}
```

**Fix: Post all, wait all**

```cpp
// ✅ Post all sends first, then wait
std::vector<MPI_Request> reqs(100);
for (int i = 0; i < 100; ++i) {
    MPI_Isend(&data[i], 1, MPI_INT, neighbor, i, comm, &reqs[i]);
}
MPI_Waitall(100, reqs.data(), MPI_STATUSES_IGNORE);
```

### Anti-Pattern 3: Unlimited Message Buffering

```cpp
// ❌ All processes send to rank 0 without waiting for receive
for (int i = 0; i < size; ++i) {
    if (rank != 0) {
        MPI_Send(data, LARGE_MSG, MPI_INT, 0, rank, comm);  // May block forever
    }
}
```

**Fix: Sender throttling**

```cpp
// ✅ Limit in-flight sends
const int MAX_INFLIGHT = 10;
std::queue<MPI_Request> pending;

for (int dest = 1; dest < size; ++dest) {
    // Drain completed requests
    while (!pending.empty()) {
        MPI_Request& req = pending.front();
        int flag;
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            pending.pop();
        } else {
            break;
        }
    }
    
    // If too many inflight, wait for one
    if (pending.size() >= MAX_INFLIGHT) {
        MPI_Request req = pending.front();
        pending.pop();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    
    // Send new request
    MPI_Request req;
    MPI_Isend(data, LARGE_MSG, MPI_INT, dest, 0, comm, &req);
    pending.push(req);
}

// Wait for remaining
while (!pending.empty()) {
    MPI_Wait(&pending.front(), MPI_STATUS_IGNORE);
    pending.pop();
}
```

---

## Production Debugging Tools

### Tool 1: MPI Error Handling Wrapper

```cpp
#define MPI_CHECK(call) \
    do { \
        int err = (call); \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int error_len; \
            MPI_Error_string(err, error_string, &error_len); \
            fprintf(stderr, "[%d] MPI Error at %s:%d: %s\n", \
                    rank, __FILE__, __LINE__, error_string); \
            MPI_Abort(comm, err); \
        } \
    } while(0)

// Usage
MPI_CHECK(MPI_Send(data, count, dtype, dest, tag, comm));
```

### Tool 2: Communication Timeline Logging

```cpp
struct CommEvent {
    double timestamp;
    const char* event_type;  // "SEND", "RECV", "BARRIER"
    int target_rank;
    int message_size;
    int tag;
};

std::vector<CommEvent> timeline;

void log_send(int dest, int size, int tag, MPI_Comm comm) {
    timeline.push_back({
        MPI_Wtime(),
        "SEND",
        dest,
        size,
        tag
    });
}

void print_timeline(int rank) {
    FILE* f = fopen(("rank_" + std::to_string(rank) + ".timeline").c_str(), "w");
    for (const auto& event : timeline) {
        fprintf(f, "%.6f %s dest=%d size=%d tag=%d\n",
               event.timestamp, event.event_type, event.target_rank, 
               event.message_size, event.tag);
    }
    fclose(f);
}
```

### Tool 3: Deadlock Detector (Requires Process Monitoring)

```bash
#!/bin/bash
# run_with_deadlock_detection.sh
#
# Usage: ./run_with_deadlock_detection.sh mpirun -np 4 ./app

timeout 30 "$@" &
MPI_PID=$!
wait $MPI_PID 2>/dev/null
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "DEADLOCK DETECTED: Application timed out"
    kill -9 $MPI_PID 2>/dev/null
    exit 1
fi

exit $EXIT_CODE
```
