# Empirical Benchmark Results & Latency Analysis

This document provides the verified experimental benchmark measurements, timing decomposition, and distributed scaling analysis for the **ESP32 Distributed INT8 Machine Learning Accelerator**.

---

## 1. Experimental Setup

The benchmark measurements were captured empirically across two physical ESP32 development modules communicating over ESP-NOW on dedicated 2.4 GHz RF Channel 1.

| Parameter | Specification | Notes |
|:---|:---|:---|
| **Microcontrollers** | 2× ESP32 Development Boards | Xtensa® Dual-Core 32-bit LX6 @ 240 MHz |
| **Node Roles** | Master (ESP32 #1), Worker (ESP32 #2) | Point-to-point symmetric MAC addressing |
| **RF Channel** | 2.4 GHz Channel 1 | Station mode (`WIFI_STA`), encryption disabled |
| **Network Protocol** | ESP-NOW (Action Frames) | Connectionless link-layer transmission |
| **Timing Precision** | Microseconds (`micros()`) | Hardware cycle-counter backed timer |
| **Arithmetic Precision**| INT8 Inputs & Weights, INT32 Accumulators | Pure integer math, no floating-point overhead |
| **Topology** | 16 → 16 → 4 Fully Connected MLP | Partitioned hidden layer (8 + 8 neurons) |

---

## 2. Verified Benchmark Data

Below are the empirical execution times recorded directly from the Master serial telemetry over repeated steady-state inference loops.

| Component | Measured Time | Category |
|:---|:---:|:---|
| **Single ESP32 inference** | **22 µs** | Computation (Baseline) |
| **Master Layer 1 (Neurons 0–7)** | **9 µs** | Computation (Local) |
| **Worker Layer 1 (Neurons 8–15)** | **13 µs** | Computation (Remote Node) |
| **Output Layer (16 → 4)** | **4 µs** | Computation (Local) |
| **Communication + Worker Turnaround** | **3754 µs** | Wireless Round-Trip + Remote Exec |
| **Verification Status** | **RESULT: CORRECT** | Parity Verification vs Reference |

---

## 3. Serial Monitor Verification

The physical test run was confirmed with the serial monitor output below:

```text
========================================
STARTING INT8 ML INFERENCE
========================================
MASTER: calculating hidden 0-7...
MASTER hidden time: 9 us
MASTER: sending input to worker...
MASTER -> WORKER: SUCCESS
WORKER RESULT RECEIVED
Worker compute time: 13 us
Communication + worker: 3754 us
Output layer time: 4 us

----------------------------------------
NEURAL NETWORK OUTPUT
----------------------------------------
Output[0] = 151
Output[1] = 137
Output[2] = 153
Output[3] = 124

Running single-ESP32 reference...
Single ESP32 inference: 22 us

========================================
ML BENCHMARK
========================================
Single ESP32: 22 us
Master Layer 1: 9 us
Worker Layer 1: 13 us
Output Layer: 4 us
Total communication + worker: 3754 us

RESULT: CORRECT
========================================
```

The captured serial interface and verification output are archived in [`images/benchmark-result.png`](../images/benchmark-result.png).

---

## 4. Latency Decomposition & Arithmetic Breakdown

To understand the system dynamics, we dissect the end-to-end turnaround time ($T_{\text{distributed}}$) into its constituent compute and communication terms:

### A. Computation Decomposition
* **Single ESP32 Reference Inference:**
  $$\text{Layer 1 (16 neurons \times 16 MACs + 16 ReLUs)} + \text{Layer 2 (4 neurons \times 16 MACs + 4 ReLUs)} = 22\ \mu\text{s}$$
* **Master Local Hidden Compute ($T_{\text{master\_L1}}$):** $9\ \mu\text{s}$ (8 neurons)
* **Worker Remote Hidden Compute ($T_{\text{worker\_L1}}$):** $13\ \mu\text{s}$ (8 neurons)
* **Master Output Compute ($T_{\text{output}}$):** $4\ \mu\text{s}$ (4 neurons)
* **Total Parallel Compute Work:**
  $$T_{\text{parallel\_compute}} = \max(T_{\text{master\_L1}}, T_{\text{worker\_L1}}) + T_{\text{output}} = \max(9, 13) + 4 = 17\ \mu\text{s}$$
  *Compared to the single-node computation baseline of $22\ \mu\text{s}$, the pure mathematical computation was reduced by $5\ \mu\text{s}$ ($22.7\%$ computational latency reduction).*

### B. Wireless Communication Overhead
The turnaround timer ($T_{\text{turnaround}} = 3754\ \mu\text{s}$) measures the delta from the moment the master invokes `esp_now_send()` until the worker's response payload is received and unpacked in `onReceive()`.

Subtracting the worker's measured computation time ($13\ \mu\text{s}$):
$$T_{\text{comm}} = T_{\text{turnaround}} - T_{\text{worker\_L1}} = 3754\ \mu\text{s} - 13\ \mu\text{s} = 3741\ \mu\text{s} \approx 3.74\ \text{ms}$$

This round-trip communication includes:
1. **Master TX preparation:** Framing `InferencePacket` into an 802.11 Vendor-Specific Action Frame.
2. **RF Transceiver Switch:** Baseband transmit FIFO loading, carrier sense, clear channel assessment (CCA), RF transmission at 2.4 GHz.
3. **Worker RX handling:** Wi-Fi ISR trigger, MAC address filtering, FreeRTOS queue dispatch to user callback.
4. **Worker TX return:** Framing `WorkerResult` and dispatching transmission.
5. **Master RX return:** Inbound frame interrupt and memory copy into activation buffer.

---

## 5. Performance Discussion & Engineering Trade-offs

### The Communication-to-Computation Asymmetry
The central finding of this research is the extreme divergence in timescales between embedded INT8 math and wireless communication:

$$\text{Computation Latency} \sim \mathcal{O}(10^1)\ \mu\text{s} \quad \text{vs.} \quad \text{Wireless Comm Latency} \sim \mathcal{O}(10^3)\ \mu\text{s}$$

The communication-to-computation ratio ($R_{\text{comm/comp}}$) for this workload is:
$$R_{\text{comm/comp}} = \frac{3741\ \mu\text{s}}{22\ \mu\text{s}} \approx 170 : 1$$

Wireless transmission overhead exceeds local execution time by two orders of magnitude. Consequently, while the partition halved the number of neurons computed per device, the total distributed wall-clock time was $3758\ \mu\text{s}$ versus $22\ \mu\text{s}$ on a single ESP32.

### Amdahl's Law in Distributed Embedded Systems
According to Amdahl’s Law, the maximum speedup achievable by parallelizing a workload across $P$ nodes is bounded by:

$$S = \frac{T_{\text{seq}}}{(1 - f) T_{\text{seq}} + \frac{f}{P} T_{\text{seq}} + T_{\text{overhead}}}$$

Where:
* $f$ is the parallelizable fraction (here $\approx \frac{18}{22} \approx 0.818$)
* $P = 2$ nodes
* $T_{\text{seq}} = 22\ \mu\text{s}$
* $T_{\text{overhead}} = T_{\text{comm}} \approx 3741\ \mu\text{s}$

When $T_{\text{overhead}} \gg T_{\text{seq}}$, the denominator is completely dominated by $T_{\text{overhead}}$, resulting in $S < 1$.

### Why Distributed Execution is Still Significant
1. **Memory Footprint Distribution:** Model weights for deep networks frequently exceed the 320 KB internal SRAM of an individual microcontroller. Partitioning layers across nodes enables running networks that physically cannot fit inside a single micro-device's memory space.
2. **Deterministic Functional Verification:** The bit-for-bit equivalence between distributed inference and reference inference proves that distributed INT8 tensor decomposition can be executed reliably over connectionless 802.11 link-layer frames without numerical drift or packet corruption.
3. **Pathway to Amortization:** The fixed round-trip transmission overhead of ~3.7 ms can be amortized when batching inferences or when evaluating much larger layer dimensions (e.g., $512 \times 512$ matrices where computation reaches tens of milliseconds).
