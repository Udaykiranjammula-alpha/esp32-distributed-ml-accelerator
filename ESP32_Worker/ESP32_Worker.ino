/**
 * @file ESP32_Worker.ino
 * @brief Distributed INT8 Machine Learning Accelerator - Worker Node (ESP32 #2)
 *
 * Description:
 * Implements the secondary compute node for the 16 -> 16 -> 4 INT8 neural network.
 * - Listens for input feature vectors via ESP-NOW from Master (ESP32 #1).
 * - Computes hidden layer partition 2 (neurons 8-15) using INT8 arithmetic with INT32 accumulation.
 * - Applies ReLU activation function.
 * - Measures compute duration with microsecond precision.
 * - Returns intermediate activations back to Master via ESP-NOW.
 */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ==============================================================================
// NEURAL NETWORK HYPERPARAMETERS & TOPOLOGY
// ==============================================================================
#define INPUT_SIZE 16        // Dimension of input feature vector (INT8)
#define HIDDEN_SIZE 16       // Total hidden layer width
#define WORKER_NEURONS 8     // Neurons assigned to this worker node (neurons 8-15)
#define OUTPUT_SIZE 4        // Dimension of final classification/regression output

// ==============================================================================
// ESP-NOW PEER CONFIGURATION
// ==============================================================================
// Target MAC address of ESP32 #1 (Master coordinator node)
uint8_t masterMAC[] = {
  0x00, 0x70, 0x07, 0x2C, 0xE5, 0x68
};

// ==============================================================================
// QUANTIZED PARAMETERS (INT8 WEIGHTS & INT32 BIASES)
// ==============================================================================
// Layer 1 weights for Worker partition: Neurons 8-15 (8 neurons x 16 inputs)
const int8_t W1_worker[WORKER_NEURONS][INPUT_SIZE] = {
  {  1, -2,  1,  2, -1,  1,  2, -2,
     1,  1, -1,  2, -2,  1,  2, -1 },

  {  2,  1, -2,  1,  2, -1,  1, -2,
     2, -1,  1,  2, -1,  1, -2,  2 },

  { -1,  2,  1, -2,  1,  2, -1,  1,
    -2,  2,  1, -1,  2, -2,  1,  2 },

  {  2, -1,  2,  1, -2,  1, -1,  2,
     1, -2,  2,  1, -1,  2, -2,  1 },

  {  1,  1,  2, -1,  2, -2,  1, -1,
     2,  1, -2,  2, -1,  1,  2, -2 },

  { -2,  1, -1,  2,  1,  2, -2,  1,
    -1,  2,  1, -2,  2, -1,  1,  2 },

  {  2, -2,  1, -1,  2,  1, -2,  2,
    -1,  1,  2, -1,  2,  1, -2,  1 },

  {  1,  2, -1,  2, -2,  1,  2, -1,
     1, -2,  1,  2, -1,  2,  1, -2 }
};

// Layer 1 biases for Worker partition: Neurons 8-15
const int32_t B1_worker[WORKER_NEURONS] = {
  1, 2, -1, 2, 1, -2, 1, 2
};

// ==============================================================================
// COMMUNICATION PACKET STRUCTURES
// ==============================================================================
// Inference packet received from Master
typedef struct {
  uint8_t type;               // Packet message type identifier (1 = Inference request)
  uint32_t sequence;          // Monotonically increasing sequence number for sync
  int8_t input[INPUT_SIZE];   // Quantized INT8 input vector
} InferencePacket;

// Result packet transmitted back to Master
typedef struct {
  uint8_t type;                         // Packet message type identifier (2 = Worker response)
  uint32_t sequence;                    // Sequence number echoing request
  int32_t hidden[WORKER_NEURONS];       // Intermediate activations computed by this worker
  uint32_t computeTimeUs;               // Execution time of worker computation in microseconds
} WorkerResult;

WorkerResult result;

// ==============================================================================
// ESP-NOW CALLBACKS
// ==============================================================================
/**
 * @brief Callback triggered when an ESP-NOW packet arrives from the Master.
 * Executes INT8 matrix multiplication for neurons 8-15, applies ReLU,
 * records execution time, and immediately transmits the response back.
 */
void onReceive(
  const esp_now_recv_info_t *info,
  const uint8_t *data,
  int len
) {

  if (len != sizeof(InferencePacket)) {
    Serial.println(
      "Invalid packet"
    );
    return;
  }

  InferencePacket packet;

  memcpy(
    &packet,
    data,
    sizeof(packet)
  );

  Serial.println();
  Serial.println(
    "WORKER: INPUT RECEIVED"
  );

  // High-resolution compute timing benchmark
  unsigned long start = micros();

  // INT8 dot product with INT32 accumulation across assigned hidden neurons
  for (
    int neuron = 0;
    neuron < WORKER_NEURONS;
    neuron++
  ) {

    int32_t sum =
      B1_worker[neuron];

    for (
      int i = 0;
      i < INPUT_SIZE;
      i++
    ) {

      sum +=
        (int32_t)packet.input[i] *
        (int32_t)W1_worker[neuron][i];
    }

    // ReLU activation: max(0, sum)
    if (sum < 0)
      sum = 0;

    result.hidden[neuron] =
      sum;
  }

  result.computeTimeUs =
    micros() - start;

  result.type = 2;

  result.sequence =
    packet.sequence;

  // Transmit intermediate activations back to Master
  esp_err_t status =
    esp_now_send(
      masterMAC,
      (uint8_t *)&result,
      sizeof(result)
    );

  Serial.print(
    "Worker compute time: "
  );

  Serial.print(
    result.computeTimeUs
  );

  Serial.println(" us");

  if (status == ESP_OK) {

    Serial.println(
      "WORKER: RESULT QUEUED"
    );

  } else {

    Serial.println(
      "WORKER: SEND FAILED"
    );
  }
}

/**
 * @brief Callback triggered upon transmission completion of an ESP-NOW frame.
 */
void onSend(
  const wifi_tx_info_t *info,
  esp_now_send_status_t status
) {

  if (
    status ==
    ESP_NOW_SEND_SUCCESS
  ) {

    Serial.println(
      "WORKER -> MASTER: SUCCESS"
    );

  } else {

    Serial.println(
      "WORKER -> MASTER: FAILED"
    );
  }
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
  Serial.println("ESP32 #2 - INT8 ML WORKER");
  Serial.println("========================================");

  Serial.println("Input: 16");
  Serial.println("Hidden neurons: 8-15");
  Serial.println("Activation: ReLU");
  Serial.println("Datatype: INT8");
  Serial.println("Accumulator: INT32");

  // Initialize ESP-NOW protocol
  if (
    esp_now_init() != ESP_OK
  ) {

    Serial.println(
      "ESP-NOW INIT FAILED"
    );

    return;
  }

  // Register communication callbacks
  esp_now_register_recv_cb(
    onReceive
  );

  esp_now_register_send_cb(
    onSend
  );

  // Configure Master peer
  esp_now_peer_info_t peerInfo = {};

  memcpy(
    peerInfo.peer_addr,
    masterMAC,
    6
  );

  peerInfo.channel = 1;
  peerInfo.encrypt = false;

  if (
    esp_now_add_peer(&peerInfo)
    != ESP_OK
  ) {

    Serial.println(
      "FAILED TO ADD MASTER"
    );

    return;
  }

  Serial.println();
  Serial.println(
    "ESP32 #2 WORKER READY"
  );
}

// ==============================================================================
// MAIN EXECUTION LOOP
// ==============================================================================
void loop() {
  // Worker operates entirely event-driven via onReceive callback
  delay(10);
}
