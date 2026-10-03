# System Architecture & Technical Specifications

This document details the hardware architecture, neural network mathematics, tensor partitioning strategy, and network protocol underlying the **ESP32 Distributed INT8 Machine Learning Accelerator**.

---

## 1. System Overview

The system establishes a distributed embedded compute cluster using two physical ESP32 microcontrollers communicating via Espressif’s connectionless **ESP-NOW** protocol.

```
                    ┌────────────────────────────────────────┐
                    │               INT8 INPUT               │
                    │           [x0, x1, ..., x15]           │
                    └───────────────────┬────────────────────┘
                                        │
                         Broadcast / Dispatch Input
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │    ESP32 #1: MASTER   │                     │    ESP32 #2: WORKER   │
     │ ───────────────────── │                     │ ───────────────────── │
     │  Layer 1 Partition 1  │       ESP-NOW       │  Layer 1 Partition 2  │
     │    Neurons 0 – 7      │ ═══════════════════ │    Neurons 8 – 15     │
     │  W1_master: 8 × 16    │      Channel 1      │  W1_worker: 8 × 16    │
     │  B1_master: 8 × 1     │                     │  B1_worker: 8 × 1     │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │                                             │
                 │ Local Activations                           │ Intermediate Activations
                 │ h[0..7]                                     │ h[8..15]
                 │                                             │ (WorkerResult Packet)
                 │                                             │
                 │              ESP-NOW Return Frame           │
                 │ ◄═══════════════════════════════════════════┘
                 ▼
     ┌───────────────────────┐
     │    ESP32 #1: MASTER   │
     │ ───────────────────── │
     │ Activation Reassembly │
     │ h[0..15] = [h1 | h2]  │
     │                       │
     │     Output Layer      │
     │     W2: 4 × 16        │
     │     B2: 4 × 1         │
     │                       │
     │ Baseline Reference &  │
     │ Parity Verification   │
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │       4 OUTPUTS       │
     │   [y0, y1, y2, y3]    │
     └───────────────────────┘
```

The system operates strictly as a distributed compute engine. It is not an IoT sensor network, smart home gadget, or telemetry gateway. Its sole purpose is distributed neural network inference and the empirical evaluation of the computation-vs-communication trade-off on edge microcontrollers.

---

## 2. Neural Network Topology & Mathematics

The network implements a Multi-Layer Perceptron (MLP) parameterized by:

$$\text{Architecture: } 16 \xrightarrow{\mathbf{W}_1, \mathbf{b}_1} 16 \xrightarrow{\text{ReLU}} 16 \xrightarrow{\mathbf{W}_2, \mathbf{b}_2} 4 \xrightarrow{\text{ReLU}} 4$$

### Mathematical Formulation

1. **Input Vector:**
   $$\mathbf{x} \in \mathbb{Z}^{16}, \quad x_i \in [-128, 127] \ (\text{INT8})$$

2. **Hidden Layer (Layer 1):**
   The hidden layer transformation performs an affine mapping followed by the Rectified Linear Unit (ReLU) activation function:
   $$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
   $$\mathbf{h} = \text{ReLU}(\mathbf{z}_1) = \max(0, \mathbf{z}_1)$$
   Where:
   * $\mathbf{W}_1 \in \mathbb{Z}^{16 \times 16}$ with weights $w_{j,i} \in \text{INT8}$
   * $\mathbf{b}_1 \in \mathbb{Z}^{16}$ with biases $b_j \in \text{INT32}$
   * $\mathbf{h} \in \mathbb{Z}^{16}$ with intermediate activations in $\text{INT32}$

3. **Output Layer (Layer 2):**
   $$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$
   $$\mathbf{y} = \text{ReLU}(\mathbf{z}_2) = \max(0, \mathbf{z}_2)$$
   Where:
   * $\mathbf{W}_2 \in \mathbb{Z}^{4 \times 16}$ with weights in $\text{INT8}$
   * $\mathbf{b}_2 \in \mathbb{Z}^{4}$ with biases in $\text{INT32}$
   * $\mathbf{y} \in \mathbb{Z}^{4}$ with output values in $\text{INT32}$

---

## 3. Distributed Tensor Partitioning Strategy

Rather than pipeline parallelism across layers, the system executes **model parallelism via row-wise weight slicing** of the first dense layer.

### Row Partitioning of Weight Matrix $\mathbf{W}_1$

The weight matrix $\mathbf{W}_1 \in \mathbb{R}^{16 \times 16}$ is horizontally sliced into two equal submatrices:

$$\mathbf{W}_1 = \begin{bmatrix} \mathbf{W}_{1,\text{master}} \\ \mathbf{W}_{1,\text{worker}} \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} \mathbf{b}_{1,\text{master}} \\ \mathbf{b}_{1,\text{worker}} \end{bmatrix}$$

* **Master Partition (Neurons 0 to 7):**
  $$\mathbf{W}_{1,\text{master}} \in \mathbb{Z}^{8 \times 16}, \quad \mathbf{b}_{1,\text{master}} \in \mathbb{Z}^{8}$$
  Computes:
  $$h_j = \max\left(0, b_{1,\text{master}}[j] + \sum_{i=0}^{15} x_i \cdot W_{1,\text{master}}[j][i]\right) \quad \text{for } j \in [0, 7]$$

* **Worker Partition (Neurons 8 to 15):**
  $$\mathbf{W}_{1,\text{worker}} \in \mathbb{Z}^{8 \times 16}, \quad \mathbf{b}_{1,\text{worker}} \in \mathbb{Z}^{8}$$
  Computes:
  $$h_{k+8} = \max\left(0, b_{1,\text{worker}}[k] + \sum_{i=0}^{15} x_i \cdot W_{1,\text{worker}}[k][i]\right) \quad \text{for } k \in [0, 7]$$

### Full Recombination & Output Layer Execution
Upon receiving the worker's activations over ESP-NOW, the Master reconstructs the full 16-element activation vector:
$$\mathbf{h} = \begin{bmatrix} \mathbf{h}_{\text{master}} \\ \mathbf{h}_{\text{worker}} \end{bmatrix} \in \mathbb{Z}^{16}$$

The Master then evaluates the final 4 output neurons locally:
$$y_m = \max\left(0, b_2[m] + \sum_{j=0}^{15} h_j \cdot W_2[m][j]\right) \quad \text{for } m \in [0, 3]$$

---

## 4. Arithmetic Precision & Overflow Prevention

Embedded neural inference must protect against integer truncation and arithmetic overflow:

1. **INT8 Multiplication:**
   Product of two INT8 integers:
   $$\text{Range: } [-128, 127] \times [-128, 127] = [-16256, 16384]$$
   This fits comfortably within a signed 16-bit word, but summing across 16 elements requires higher dynamic range.

2. **INT32 Accumulator:**
   The dot product loop accumulates into a signed 32-bit integer register:
   ```cpp
   int32_t sum = B1_master[neuron];
   for (int i = 0; i < INPUT_SIZE; i++) {
     sum += (int32_t)packet.input[i] * (int32_t)W1_master[neuron][i];
   }
   ```
   * Maximum theoretical accumulated value: $2 \times 10^9$, far below INT32 maximum ($2,147,483,647$).
   * Zero risk of numerical overflow without necessitating slow floating-point emulation.

3. **Non-Linearity (ReLU):**
   Executed directly in integer domain:
   ```cpp
   if (sum < 0) sum = 0;
   ```

---

## 5. ESP-NOW Network Protocol & Packet Structures

ESP-NOW transmits raw IEEE 802.11 Vendor-Specific Action Frames at Layer 2 (Data Link layer). It bypasses the IP layer, TCP/UDP overhead, and Wi-Fi handshake states.

### Packet 1: `InferencePacket` (Master $\to$ Worker)
Carries the 16-element input vector to the worker.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Type (0x01)  |                   Sequence                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| (Seq cont'd)  |   input[0]    |   input[1]    |   input[2]    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   input[3]    |   input[4]    |   input[5]    |   input[6]    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   input[7]    |   input[8]    |   input[9]    |   input[10]   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   input[11]   |   input[12]   |   input[13]   |   input[14]   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   input[15]   |
+-+-+-+-+-+-+-+-+
```

| Field | Type | Size | Description |
|:---|:---|:---:|:---|
| `type` | `uint8_t` | 1 byte | Message ID (`0x01` = Inference Request) |
| `sequence` | `uint32_t` | 4 bytes | Monotonic request counter for synchronization |
| `input` | `int8_t[16]` | 16 bytes | Quantized INT8 input feature vector |
| **Total Size** | | **21 bytes** | Compact payload |

---

### Packet 2: `WorkerResult` (Worker $\to$ Master)
Returns the computed activations for neurons 8–15 along with worker-side telemetry.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Type (0x02)  |                   Sequence                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| (Seq cont'd)  |                   hidden[0]                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   hidden[1]                   |   hidden[2]   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| (h[2] cont'd) |                   hidden[3]                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   hidden[4]                   |   hidden[5]   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| (h[5] cont'd) |                   hidden[6]                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   hidden[7]                   | computeTimeUs |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| (time cont'd) |
+-+-+-+-+-+-+-+-+
```

| Field | Type | Size | Description |
|:---|:---|:---:|:---|
| `type` | `uint8_t` | 1 byte | Message ID (`0x02` = Result Return) |
| `sequence` | `uint32_t` | 4 bytes | Echo of sequence number |
| `hidden` | `int32_t[8]` | 32 bytes | Intermediate activations for neurons 8–15 |
| `computeTimeUs`| `uint32_t` | 4 bytes | Measured compute latency on the Worker node |
| **Total Size** | | **41 bytes** | Direct packed memory payload |

---

## 6. Execution Flow & Lifecycle Sequence

```
ESP32 #1 (Master)                                         ESP32 #2 (Worker)
       │                                                         │
   [Generate Input Vector]                                       │
       │                                                         │
   [Compute Hidden 0-7 (9 µs)]                                   │
       │                                                         │
       ├─────────── ESP-NOW: InferencePacket (21 B) ────────────►│
       │                                                         │
       │                                                [Wi-Fi RX Interrupt]
       │                                                         │
       │                                                [Compute Hidden 8-15 (13 µs)]
       │                                                         │
       │◄────────── ESP-NOW: WorkerResult (41 B) ────────────────┤
       │                                                         │
   [Reassemble Hidden Buffer h[0..15]]                           │
       │                                                         │
   [Compute Output Layer (4 µs)]                                 │
       │                                                         │
   [Run Local Reference Baseline (22 µs)]                        │
       │                                                         │
   [Bit-for-bit Parity Verification]                             │
       │                                                         │
   [Print Telemetry & RESULT: CORRECT]                           │
```

1. **Input Generation:** Synthetic INT8 feature vector `packet.input[i] = (i % 7) - 3` is synthesized on the Master.
2. **Local Hidden Execution:** Master iterates over neurons 0–7, performing dot products and ReLU activation ($9\ \mu\text{s}$).
3. **ESP-NOW Dispatch:** Master sends `InferencePacket` to Worker MAC address.
4. **Worker Hidden Execution:** Worker receives packet in `onReceive()`, computes neurons 8–15 ($13\ \mu\text{s}$), and queues `WorkerResult` back to Master.
5. **Recombination:** Master receives `WorkerResult`, copies `hidden[0..7]` from the payload into `hidden[8..15]`.
6. **Output Execution:** Master multiplies the combined 16-element vector by `W2[4][16]` ($4\ \mu\text{s}$).
7. **Verification:** Master calculates the entire 16-neuron inference independently and confirms bit-exact equivalence (`RESULT: CORRECT`).
