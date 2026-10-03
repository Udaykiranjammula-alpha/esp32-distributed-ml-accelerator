/**
 * @file ESP32_Master.ino
 * @brief Distributed INT8 Machine Learning Accelerator - Master Node (ESP32 #1)
 *
 * Description:
 * Implements the coordinator and compute node for a 16 -> 16 -> 4 INT8 neural network.
 * - Computes hidden layer partition 1 (neurons 0-7) locally.
 * - Offloads hidden layer partition 2 (neurons 8-15) to ESP32 #2 via ESP-NOW.
 * - Reassembles intermediate activations and computes the final output layer (4 neurons).
 * - Executes an independent single-node reference inference for verification and benchmarking.
 */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ==============================================================================
// NEURAL NETWORK HYPERPARAMETERS & TOPOLOGY
// ==============================================================================
#define INPUT_SIZE 16        // Dimension of input feature vector (INT8)
#define HIDDEN_SIZE 16       // Total hidden layer width
#define WORKER_NEURONS 8     // Neurons assigned per node (Master: 0-7, Worker: 8-15)
#define OUTPUT_SIZE 4        // Dimension of final classification/regression output

// ==============================================================================
// ESP-NOW PEER CONFIGURATION
// ==============================================================================
// Target MAC address of ESP32 #2 (Worker node)
uint8_t workerMAC[] = {
  0x00, 0x70, 0x07, 0xE2, 0x16, 0xF0
};

// ==============================================================================
// QUANTIZED PARAMETERS (INT8 WEIGHTS & INT32 BIASES)
// ==============================================================================
// Layer 1 weights for Master partition: Neurons 0-7 (8 neurons x 16 inputs)
const int8_t W1_master[WORKER_NEURONS][INPUT_SIZE] = {
  {  1, -2,  1,  2, -1,  1,  2, -2,  1,  1, -1,  2, -2,  1,  2, -1 },
  {  2,  1, -2,  1,  2, -1,  1, -2,  2, -1,  1,  2, -1,  1, -2,  2 },
  { -1,  2,  1, -2,  1,  2, -1,  1, -2,  2,  1, -1,  2, -2,  1,  2 },
  {  2, -1,  2,  1, -2,  1, -1,  2,  1, -2,  2,  1, -1,  2, -2,  1 },
  {  1,  1,  2, -1,  2, -2,  1, -1,  2,  1, -2,  2, -1,  1,  2, -2 },
  { -2,  1, -1,  2,  1,  2, -2,  1, -1,  2,  1, -2,  2, -1,  1,  2 },
  {  2, -2,  1, -1,  2,  1, -2,  2, -1,  1,  2, -1,  2,  1, -2,  1 },
  {  1,  2, -1,  2, -2,  1,  2, -1,  1, -2,  1,  2, -1,  2,  1, -2 }
};

// Layer 1 biases for Master partition: Neurons 0-7
const int32_t B1_master[WORKER_NEURONS] = {
  1, 2, -1, 2, 1, -2, 1, 2
};

// Layer 2 weights (Output layer): 4 output neurons x 16 full hidden activations
const int8_t W2[OUTPUT_SIZE][HIDDEN_SIZE] = {
  {
     1, -1,  2,  1,
    -2,  1,  2, -1,
     1,  2, -1,  1,
     2, -2,  1,  2
  },
  {
    -1,  2,  1, -2,
     1,  2, -1,  1,
    -2,  1,  2, -1,
     1,  2, -2,  1
  },
  {
     2,  1, -1,  1,
     1, -2,  2, -1,
     2,  1, -2,  1,
    -1,  2,  1, -2
  },
  {
     1,  2, -2,  1,
     2, -1,  1,  2,
    -1,  1,  2, -2,
     1, -1,  2,  1
  }
};

// Layer 2 biases (Output layer)
const int32_t B2[OUTPUT_SIZE] = {
  1, 2, 1, -1
};

// ==============================================================================
// COMMUNICATION PACKET STRUCTURES
// ==============================================================================
// Inference packet transmitted from Master to Worker
typedef struct {
  uint8_t type;               // Packet message type identifier (1 = Inference request)
  uint32_t sequence;          // Monotonically increasing sequence number for sync
  int8_t input[INPUT_SIZE];   // Quantized INT8 input vector
} InferencePacket;

// Result packet received by Master from Worker
typedef struct {
  uint8_t type;                         // Packet message type identifier (2 = Worker response)
  uint32_t sequence;                    // Sequence number echoing request
  int32_t hidden[WORKER_NEURONS];       // Computed intermediate activations (neurons 8-15)
  uint32_t computeTimeUs;               // Execution time of worker computation in microseconds
} WorkerResult;

// State and buffer variables
InferencePacket packet;
volatile bool workerReady = false;
WorkerResult workerResult;

int32_t hidden[HIDDEN_SIZE];             // Aggregated hidden activation buffer (0-15)
int32_t output[OUTPUT_SIZE];             // Output buffer for distributed inference
int32_t referenceOutput[OUTPUT_SIZE];    // Output buffer for single-node reference inference

uint32_t sequenceNumber = 0;

// ==============================================================================
// ESP-NOW CALLBACKS
// ==============================================================================
/**
 * @brief Callback triggered when an ESP-NOW packet is received.
 * Assembles intermediate hidden activations from the Worker into hidden[8..15].
 */
void onReceive(
  const esp_now_recv_info_t *info,
  const uint8_t *data,
  int len
) {
  if (len != sizeof(WorkerResult)) {
    Serial.println("Invalid worker result");
    return;
  }

  memcpy(&workerResult, data, sizeof(workerResult));

  // Merge worker activations into the upper partition of the hidden layer buffer
  for (int i = 0; i < WORKER_NEURONS; i++) {
    hidden[WORKER_NEURONS + i] = workerResult.hidden[i];
  }

  workerReady = true;
}

/**
 * @brief Callback triggered upon transmission completion of an ESP-NOW frame.
 */
void onSend(
  const wifi_tx_info_t *info,
  esp_now_send_status_t status
) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    Serial.println("MASTER -> WORKER: SUCCESS");
  } else {
    Serial.println("MASTER -> WORKER: FAILED");
  }
}

// ==============================================================================
// NEURAL NETWORK COMPUTATION ROUTINES
// ==============================================================================
/**
 * @brief Computes hidden neurons 0-7 locally on the Master node.
 * Uses INT8 multiplication with INT32 accumulation, followed by ReLU activation.
 * @return Execution time in microseconds.
 */
unsigned long calculateMasterHidden() {

  unsigned long start = micros();

  for (int neuron = 0; neuron < WORKER_NEURONS; neuron++) {

    int32_t sum = B1_master[neuron];

    for (int i = 0; i < INPUT_SIZE; i++) {
      sum +=
        (int32_t)packet.input[i] *
        (int32_t)W1_master[neuron][i];
    }

    // ReLU activation: max(0, sum)
    if (sum < 0)
      sum = 0;

    hidden[neuron] = sum;
  }

  return micros() - start;
}

/**
 * @brief Computes the output layer (16 hidden inputs -> 4 outputs) on the Master node.
 * Evaluated once both local and worker hidden activations are assembled.
 * @return Execution time in microseconds.
 */
unsigned long calculateOutput() {

  unsigned long start = micros();

  for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) {

    int32_t sum = B2[neuron];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
      sum += hidden[i] * W2[neuron][i];
    }

    // ReLU activation: max(0, sum)
    if (sum < 0)
      sum = 0;

    output[neuron] = sum;
  }

  return micros() - start;
}

/**
 * @brief Computes end-to-end inference entirely on the Master node as a baseline reference.
 * Uses identical parameters for verification and execution-time benchmarking.
 * @return Execution time in microseconds.
 */
unsigned long calculateReference() {

  int32_t refHidden[HIDDEN_SIZE];

  unsigned long start = micros();

  // Full 16-neuron hidden layer computation
  for (int neuron = 0; neuron < HIDDEN_SIZE; neuron++) {

    int32_t sum;

    if (neuron < WORKER_NEURONS) {

      sum = B1_master[neuron];

      for (int i = 0; i < INPUT_SIZE; i++) {
        sum +=
          (int32_t)packet.input[i] *
          (int32_t)W1_master[neuron][i];
      }

    } else {

      int workerNeuron =
        neuron - WORKER_NEURONS;

      sum = B1_master[workerNeuron];

      for (int i = 0; i < INPUT_SIZE; i++) {
        sum +=
          (int32_t)packet.input[i] *
          (int32_t)W1_master[workerNeuron][i];
      }
    }

    // ReLU activation
    if (sum < 0)
      sum = 0;

    refHidden[neuron] = sum;
  }

  // Output layer computation
  for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) {

    int32_t sum = B2[neuron];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
      sum += refHidden[i] * W2[neuron][i];
    }

    // ReLU activation
    if (sum < 0)
      sum = 0;

    referenceOutput[neuron] = sum;
  }

  return micros() - start;
}

/**
 * @brief Prints final output vector to the Serial Monitor.
 */
void printOutput() {

  Serial.println();
  Serial.println("----------------------------------------");
  Serial.println("NEURAL NETWORK OUTPUT");
  Serial.println("----------------------------------------");

  for (int i = 0; i < OUTPUT_SIZE; i++) {

    Serial.print("Output[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(output[i]);
  }
}

/**
 * @brief Compares distributed execution output against single-ESP32 reference inference.
 * @return true if all outputs match bit-for-bit, false otherwise.
 */
bool verify() {

  for (int i = 0; i < OUTPUT_SIZE; i++) {

    if (output[i] != referenceOutput[i]) {

      Serial.println("VERIFICATION FAILED");

      Serial.print("Output ");
      Serial.print(i);
      Serial.print(" expected=");
      Serial.print(referenceOutput[i]);
      Serial.print(" actual=");
      Serial.println(output[i]);

      return false;
    }
  }

  return true;
}

// ==============================================================================
// SETUP & INITIALIZATION
// ==============================================================================
void setup() {

  Serial.begin(115200);

  delay(1000);

  // Set Wi-Fi station mode and lock to Channel 1
  WiFi.mode(WIFI_STA);

  esp_wifi_set_channel(
    1,
    WIFI_SECOND_CHAN_NONE
  );

  Serial.println();
  Serial.println("========================================");
  Serial.println("ESP32 #1 - DISTRIBUTED INT8 ML MASTER");
  Serial.println("========================================");

  Serial.println("Network: 16 -> 16 -> 4");
  Serial.println("Activation: ReLU");
  Serial.println("Input: INT8");
  Serial.println("Accumulator: INT32");

  // Initialize ESP-NOW protocol
  if (esp_now_init() != ESP_OK) {

    Serial.println("ESP-NOW INIT FAILED");

    return;
  }

  // Register communication callbacks
  esp_now_register_recv_cb(onReceive);
  esp_now_register_send_cb(onSend);

  // Configure Worker peer
  esp_now_peer_info_t peerInfo = {};

  memcpy(
    peerInfo.peer_addr,
    workerMAC,
    6
  );

  peerInfo.channel = 1;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {

    Serial.println("FAILED TO ADD WORKER");

    return;
  }

  Serial.println();
  Serial.println("MASTER READY");
}

// ==============================================================================
// MAIN EXECUTION LOOP
// ==============================================================================
void loop() {

  Serial.println();
  Serial.println("========================================");
  Serial.println("STARTING INT8 ML INFERENCE");
  Serial.println("========================================");

  sequenceNumber++;

  // Generate synthetic input vector for testing
  for (int i = 0; i < INPUT_SIZE; i++) {
    packet.input[i] = (i % 7) - 3;
  }

  packet.type = 1;
  packet.sequence = sequenceNumber;

  workerReady = false;

  // 1. Calculate Master hidden partition (neurons 0-7)
  Serial.println(
    "MASTER: calculating hidden 0-7..."
  );

  unsigned long masterTime =
    calculateMasterHidden();

  Serial.print("MASTER hidden time: ");
  Serial.print(masterTime);
  Serial.println(" us");

  // 2. Offload inputs to Worker node via ESP-NOW
  Serial.println(
    "MASTER: sending input to worker..."
  );

  unsigned long communicationStart =
    micros();

  esp_err_t status = esp_now_send(
    workerMAC,
    (uint8_t *)&packet,
    sizeof(packet)
  );

  if (status != ESP_OK) {

    Serial.println("INPUT SEND FAILED");

    delay(3000);

    return;
  }

  // 3. Await worker response with 5-second safety timeout
  unsigned long timeout =
    millis() + 5000;

  while (
    !workerReady &&
    millis() < timeout
  ) {
    delay(1);
  }

  unsigned long communicationTime =
    micros() - communicationStart;

  if (!workerReady) {

    Serial.println();
    Serial.println("WORKER TIMEOUT");

    delay(3000);

    return;
  }

  Serial.println(
    "WORKER RESULT RECEIVED"
  );

  Serial.print("Worker compute time: ");
  Serial.print(workerResult.computeTimeUs);
  Serial.println(" us");

  Serial.print("Communication + worker: ");
  Serial.print(communicationTime);
  Serial.println(" us");

  // 4. Compute final output layer
  unsigned long outputTime =
    calculateOutput();

  Serial.print("Output layer time: ");
  Serial.print(outputTime);
  Serial.println(" us");

  printOutput();

  // 5. Compute single-ESP32 reference baseline
  Serial.println();
  Serial.println(
    "Running single-ESP32 reference..."
  );

  unsigned long referenceTime =
    calculateReference();

  Serial.print("Single ESP32 inference: ");
  Serial.print(referenceTime);
  Serial.println(" us");

  // 6. Verify correctness
  bool correct = verify();

  // 7. Output benchmark telemetry
  Serial.println();
  Serial.println("========================================");
  Serial.println("ML BENCHMARK");
  Serial.println("========================================");

  Serial.print("Single ESP32: ");
  Serial.print(referenceTime);
  Serial.println(" us");

  Serial.print("Master Layer 1: ");
  Serial.print(masterTime);
  Serial.println(" us");

  Serial.print("Worker Layer 1: ");
  Serial.print(workerResult.computeTimeUs);
  Serial.println(" us");

  Serial.print("Output Layer: ");
  Serial.print(outputTime);
  Serial.println(" us");

  Serial.print(
    "Total communication + worker: "
  );

  Serial.print(communicationTime);
  Serial.println(" us");

  Serial.println();

  if (correct) {
    Serial.println("RESULT: CORRECT");
  } else {
    Serial.println("RESULT: FAILED");
  }

  Serial.println(
    "========================================"
  );

  delay(5000);
}
