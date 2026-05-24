---
name: mpi-best-practices
description: Develop correct, efficient, and deadlock-free MPI applications. Use this skill whenever working with MPI (Message Passing Interface) code — whether writing new MPI programs, debugging communication issues, avoiding deadlocks, optimizing collective operations, reducing network congestion, or implementing proper send/receive protocols. Triggers on MPI code reviews, MPI development, parallel programming with message passing, HPC application development, MPI communication patterns, collective operations, non-blocking communications, and performance tuning of distributed memory systems.
compatibility: Requires understanding of C, C++, or Fortran; knowledge of distributed memory parallel computing concepts beneficial
---

# MPI Best Practices: Protocol Correctness, Deadlock Avoidance, and Congestion Prevention

This skill guides safe, correct, and efficient MPI application development by preventing three classes of errors: **protocol violations, deadlocks, and communication bottlenecks**.

## Core Principles

### 1. Protocol Correctness

MPI communication protocols are strict contracts. Violating them causes hangs, data corruption, or crashes.

**Sender-Receiver Matching Rules:**
- Each `MPI_Send` must match exactly one `MPI_Recv` on the target
- Rank, tag, and communicator must match on both sides
- Blocking sends assume a receiver exists; non-blocking sends do not
- Wildcard receives (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`) are valid but must match available messages

**Type Safety:**
- Send and receive datatypes must be compatible
- `MPI_Pack` / `MPI_Unpack` for heterogeneous data
- Never assume integer sizes or floating-point representations match across machines

### 2. Deadlock Avoidance

Deadlock occurs when processes wait indefinitely for events that never occur. It's the most common failure mode in MPI.

**Circular Wait Patterns:**
```
Process 0: Send to 1 → Recv from 1  ❌ If 1 tries to send to 0 first, both block
Process 1: Send to 0 → Recv from 0
```

**The Critical Problem:**
- Limited buffering in MPI implementations (typically ~64-256 KB system buffers)
- `MPI_Send` is NOT guaranteed to be non-blocking even with `MPI_Send` (may block if buffer fills)
- Using only `MPI_Send` + `MPI_Recv` in symmetric patterns guarantees deadlock on large messages

### 3. Communication Congestion

Network congestion reduces throughput, increases latency, and amplifies deadlock risk.

**Congestion Sources:**
- All-to-all operations on large process counts
- Unbalanced message sizes (some processes send large messages, others small)
- Synchronous waits blocking network progress
- Insufficient overlap between computation and communication

## Recipe 1: Safe Point-to-Point Communication

### Pattern A: Producer-Consumer (Hierarchical)

**Use when:** One rank sends to many, or many send to one. Avoids circular waits.

```cpp
#include <mpi.h>
#include <vector>

void send_to_consumer(int rank, int root, MPI_Datatype dtype, int count) {
    if (rank != root) {
        // Non-root: send data to root
        MPI_Ssend(data, count, dtype, root, 0, MPI_COMM_WORLD);
    } else {
        // Root: receive from all others
        std::vector<MPI_Request> requests;
        for (int src = 0; src < MPI_Comm_size; ++src) {
            if (src != root) {
                MPI_Request req;
                MPI_Irecv(buffer[src], count, dtype, src, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
}
```

**Why it works:**
- Non-root ranks are active (sending), root is passive (receiving)
- No circular dependency: senders don't wait for replies
- Use `MPI_Ssend` (synchronous send) to block until receiver is ready — prevents buffer overflow

### Pattern B: Bidirectional Exchange (Non-blocking)

**Use when:** Ranks need to exchange data bilaterally.

```cpp
void safe_bidirectional_exchange(int rank, int size, int* send_buf, int* recv_buf, 
                                 int count, MPI_Comm comm) {
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    
    MPI_Request req[4];
    
    // Post all receives first (non-blocking)
    MPI_Irecv(recv_buf, count, MPI_INT, left, 0, comm, &req[0]);
    MPI_Irecv(recv_buf + count, count, MPI_INT, right, 1, comm, &req[1]);
    
    // Then post all sends (non-blocking)
    MPI_Isend(send_buf, count, MPI_INT, left, 1, comm, &req[2]);
    MPI_Isend(send_buf + count, count, MPI_INT, right, 0, comm, &req[3]);
    
    // Wait for all to complete
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}
```

**Why it works:**
- Non-blocking operations posted first, then waited on later
- Allows MPI progress (background communication) while rank is not blocked
- No circular wait because all sends are non-blocking
- Different tags (`0` vs `1`) prevent message mismatches

### Pattern C: Ring Reduction (Pipelined)

**Use when:** Aggregating data across a chain of processes while avoiding congestion.

```cpp
void ring_reduction(int rank, int size, int* data, int count, MPI_Op op, MPI_Comm comm) {
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    
    // Pipeline: process chunk i at rank i % size
    for (int step = 0; step < size - 1; ++step) {
        int send_rank = (rank - step - 1 + size) % size;
        int recv_rank = (rank + step + 1) % size;
        
        int send_buf[count], recv_buf[count];
        
        // Use irecv/isend to avoid blocking
        MPI_Request req[2];
        MPI_Irecv(recv_buf, count, MPI_INT, recv_rank, step, comm, &req[0]);
        MPI_Isend(data, count, MPI_INT, send_rank, step, comm, &req[1]);
        
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // Wait for receive only
        
        // Apply operation locally
        for (int i = 0; i < count; ++i) {
            if (op == MPI_SUM) data[i] += recv_buf[i];
            // Handle other ops similarly
        }
        
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);  // Ensure send completes
    }
}
```

**Why it works:**
- Message flow is unidirectional (no cycles)
- Each step processes one chunk, reducing congestion
- Pipelining hides latency

---

## Recipe 2: Deadlock-Free Collective Operations

### Standard Collectives (Already Deadlock-Free)

```cpp
// These are safe by design — all ranks call them in same order
MPI_Bcast(data, count, dtype, root, comm);
MPI_Allreduce(sendbuf, recvbuf, count, dtype, op, comm);
MPI_Alltoall(sendbuf, sendcount, dtype, recvbuf, recvcount, dtype, comm);
MPI_Barrier(comm);
```

**Key rule:** All ranks in the communicator must call the same collective in the same order.

### Conditional Collectives (⚠️ Dangerous)

```cpp
// ❌ DEADLOCK RISK: not all ranks execute the collective
if (rank % 2 == 0) {
    MPI_Allreduce(a, b, 1, MPI_INT, MPI_SUM, comm);  // Only even ranks call this
}
```

**Safe pattern: Use subcommunicators**

```cpp
MPI_Comm even_comm, odd_comm;

// Create subcommunicators
int color = rank % 2;
MPI_Comm_split(comm, color, rank, &new_comm);

// Now all ranks in new_comm call the collective
MPI_Allreduce(a, b, 1, MPI_INT, MPI_SUM, new_comm);

MPI_Comm_free(&new_comm);
```

### Non-blocking Collectives (MPI 3.0+)

```cpp
MPI_Request req;
MPI_Iallreduce(sendbuf, recvbuf, count, dtype, op, comm, &req);
// Do computation while collective happens in background
do_useful_work();
MPI_Wait(&req, MPI_STATUS_IGNORE);  // Wait when result needed
```

**Advantage:** Hides collective latency with overlap.

---

## Recipe 3: Congestion-Aware Communication

### Anti-Pattern: Synchronized All-to-All

```cpp
// ❌ All processes send to rank 0 in lockstep — network swamp!
for (int i = 0; i < size; ++i) {
    if (rank == 0) {
        for (int src = 1; src < size; ++src) {
            MPI_Recv(buffer, count, dtype, src, 0, comm, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(buffer, count, dtype, 0, 0, comm);
    }
}
```

### Pattern: Asynchronous All-to-All with Message Ordering

```cpp
void congestion_aware_gather(int rank, int size, int* sendbuf, int* recvbuf, 
                            int count, MPI_Datatype dtype, MPI_Comm comm) {
    if (rank == 0) {
        // Root: post receives in staggered fashion
        std::vector<MPI_Request> reqs;
        for (int src = 1; src < size; ++src) {
            MPI_Request req;
            // Stagger receives to prevent buffer buildup
            MPI_Irecv(recvbuf + src * count, count, dtype, src, src, comm, &req);
            reqs.push_back(req);
        }
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    } else {
        // Non-roots: send with rank as tag (allows root to order receives)
        MPI_Send(sendbuf, count, dtype, 0, rank, comm);
    }
}
```

### Pattern: Pipeline Large All-to-All

```cpp
void pipelined_alltoall(int rank, int size, int* sendbuf, int* recvbuf, 
                       int count, MPI_Comm comm) {
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    
    // Send to right, receive from left in ring
    for (int step = 0; step < size; ++step) {
        int send_to = (rank + step) % size;
        int recv_from = (rank - step + size) % size;
        
        MPI_Sendrecv(sendbuf + send_to * count, count, MPI_INT,
                    send_to, 0,
                    recvbuf + recv_from * count, count, MPI_INT,
                    recv_from, 0,
                    comm, MPI_STATUS_IGNORE);
    }
}
```

**Why it works:**
- `MPI_Sendrecv` guarantees no deadlock (implementation handles buffering)
- Ring topology limits per-rank congestion
- Scales better than fully-connected all-to-all for large process counts

---

## Recipe 4: Debugging Deadlock

### Symptom: Application Hangs

**Step 1: Check Process State (in debugger or timeout handler)**

```cpp
// Add a timeout wrapper
int timeout_secs = 30;
MPI_Request req;
MPI_Ibarrier(comm, &req);

int flag = 0;
for (int t = 0; t < timeout_secs; ++t) {
    sleep(1);
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    if (flag) break;
}

if (!flag) {
    fprintf(stderr, "DEADLOCK: Rank %d timed out at barrier\n", rank);
    fflush(stderr);
    MPI_Abort(comm, 1);
}
```

### Step 2: Identify Symmetry Violations

Check:
- Are all ranks calling the same collectives in the same order?
- Are all point-to-point sends matched by receives?
- Are there conditional collectives on subset of ranks?

### Step 3: Add Communication Tracing

```cpp
#define TRACE_COMM(fmt, ...) \
    do { fprintf(stderr, "[%d] " fmt "\n", rank, ##__VA_ARGS__); fflush(stderr); } while(0)

TRACE_COMM("About to send to rank %d", target);
MPI_Send(data, count, dtype, target, tag, comm);
TRACE_COMM("Send to rank %d completed", target);
```

**Look for:**
- Send completed but receive never starts → missing receive
- Receive waiting but send never completes → message mismatch (wrong tag/rank)
- Both waiting → circular dependency

### Step 4: Force Progress with MPI_Iprobe

```cpp
// Non-blocking check for pending messages
int flag;
MPI_Status status;
MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
if (flag) {
    printf("Rank %d has pending message from %d with tag %d\n", 
           rank, status.MPI_SOURCE, status.MPI_TAG);
}
```

---

## Recipe 5: Performance Profiling for Congestion Detection

### Memory Bandwidth Benchmark

```cpp
void bandwidth_test(int rank, int size, MPI_Comm comm) {
    int iterations = 100;
    int msg_size = 1024 * 1024;  // 1 MB
    std::vector<int> buffer(msg_size);
    
    if (rank == 0) {
        auto start = MPI_Wtime();
        for (int i = 0; i < iterations; ++i) {
            MPI_Send(buffer.data(), msg_size, MPI_INT, 1, 0, comm);
            MPI_Recv(buffer.data(), msg_size, MPI_INT, 1, 0, comm, MPI_STATUS_IGNORE);
        }
        auto elapsed = MPI_Wtime() - start;
        double bandwidth = (2.0 * msg_size * iterations * 8) / (elapsed * 1e9);
        printf("Bandwidth: %.2f Gbps\n", bandwidth);
    } else if (rank == 1) {
        for (int i = 0; i < iterations; ++i) {
            MPI_Recv(buffer.data(), msg_size, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Send(buffer.data(), msg_size, MPI_INT, 0, 0, comm);
        }
    }
}
```

### Collective Communication Profiling

```cpp
double time_collective(MPI_Comm comm, int iterations) {
    MPI_Barrier(comm);
    auto start = MPI_Wtime();
    for (int i = 0; i < iterations; ++i) {
        MPI_Allreduce(MPI_IN_PLACE, nullptr, 1, MPI_INT, MPI_SUM, comm);
    }
    MPI_Barrier(comm);
    return MPI_Wtime() - start;
}
```

**Interpret results:**
- If time grows superlinearly with process count → congestion
- If large messages slower than small by < 2x → good bandwidth utilization
- If barrier time > 10ms on < 1000 processes → congestion or synchronization overhead

---

## Checklist: Before Shipping MPI Code

- [ ] **Protocol**: All `MPI_Send` calls matched by `MPI_Recv` with correct rank/tag
- [ ] **Type safety**: Send and receive types are compatible
- [ ] **Non-blocking used**: High-risk patterns (all-to-all, ring communications) use `MPI_I*` operations
- [ ] **Circular waits eliminated**: No process waits for response from process it's sending to
- [ ] **Symmetric collectives**: All ranks in communicator call same collectives in same order
- [ ] **Subcommunicators for conditionals**: Conditional collectives use `MPI_Comm_split` or subsets
- [ ] **Error handling**: `MPI_Abort` on critical errors, don't silently ignore failures
- [ ] **Congestion mitigated**: Large all-to-all uses pipelined or ring topology, not full-mesh
- [ ] **Progress ensured**: Long-running code has periodic `MPI_Iprobe` or progress checks
- [ ] **Tested at scale**: Deadlock tests run with process counts >> development machine

---

## References

- **MPI 3.1 Standard**: [https://www.mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
- **MPICH Implementation**: [https://www.mpich.org/documentation/](https://www.mpich.org/documentation/)
- **OpenMPI Docs**: [https://www.open-mpi.org/doc/](https://www.open-mpi.org/doc/)

---

## Quick Decision Tree

```
Is your code hanging?
├─ Yes, at barrier/collective?
│  ├─ Not all ranks reached it? → Conditional collective. Use MPI_Comm_split
│  └─ All ranks there? → Check network/node availability
├─ Yes, at send/recv?
│  ├─ Same rank trying to recv from itself? → Tag mismatch
│  └─ Ring pattern (A→B, B→A)? → Use non-blocking + wait, or MPI_Sendrecv
└─ No, but slow?
   ├─ All-to-all communication? → Use ring topology instead
   └─ Large messages? → Profile bandwidth, check network saturation
```
