// ESP32-C3-MINI-1-H4 <-> ADS1298 + LSM6DSOX BLE binary streamer with RLD.
// Select ESP32C3 Dev Module with USB CDC On Boot enabled.

#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <SPI.h>

#include "lsm6dsox_spi.h"

constexpr uint8_t PIN_MOSI = 1;     // U4 pad 13, GPIO1 -> ADS DIN
constexpr uint8_t PIN_MISO = 0;     // U4 pad 12, GPIO0 <- ADS DOUT
constexpr uint8_t PIN_CS_IMU = 7;   // U4 pad 21, GPIO7
constexpr uint8_t PIN_DRDY = 4;
constexpr uint8_t PIN_RESET = 5;
constexpr uint8_t PIN_CS_ADS = 3;   // U4 pad 6, GPIO3
constexpr uint8_t PIN_SCLK = 6;     // U4 pad 20, GPIO6
constexpr uint8_t PIN_PWDN = 10;

constexpr uint8_t CMD_START = 0x08;
constexpr uint8_t CMD_RDATAC = 0x10;
constexpr uint8_t CMD_SDATAC = 0x11;
constexpr uint8_t CMD_RREG = 0x20;
constexpr uint8_t CMD_WREG = 0x40;

constexpr uint8_t REG_ID = 0x00;
constexpr uint8_t REG_CONFIG1 = 0x01;
constexpr uint8_t REG_CONFIG2 = 0x02;
constexpr uint8_t REG_CONFIG3 = 0x03;
constexpr uint8_t REG_CH1SET = 0x05;
constexpr uint8_t REG_RLD_SENSP = 0x0D;
constexpr uint8_t REG_RLD_SENSN = 0x0E;

constexpr uint8_t ADS1298_ID = 0x92;
constexpr uint8_t CONFIG1_500_SPS_HR = 0x86;
constexpr uint8_t CONFIG2_TEST_OFF = 0x00;
constexpr uint8_t CONFIG3_RLD_ON_INTERNAL_REF = 0xCC;
constexpr uint8_t CHANNEL_NORMAL_GAIN6 = 0x00;
constexpr uint8_t RLD_SENSP_CH1 = 0x01;
constexpr uint8_t RLD_SENSN_CH1 = 0x01;

constexpr uint8_t ADS_FRAME_BYTES = 27;
constexpr uint8_t ADS_CHANNEL_COUNT = 8;
constexpr size_t SAMPLE_QUEUE_CAPACITY = 128;
constexpr size_t IMU_QUEUE_CAPACITY = 32;

constexpr char BLE_DEVICE_NAME[] = "SensorBiShe-EMG";
constexpr char BLE_SERVICE_UUID[] =
    "7b1e0001-6f9b-4c3d-8a6e-2f4a5b6c7d80";
constexpr char BLE_DATA_CHARACTERISTIC_UUID[] =
    "7b1e0002-6f9b-4c3d-8a6e-2f4a5b6c7d80";
constexpr uint8_t BLE_PACKET_MAGIC_0 = 0xA5;
constexpr uint8_t BLE_PACKET_MAGIC_1 = 0x5A;
constexpr uint8_t BLE_PROTOCOL_VERSION = 2;
constexpr uint8_t BLE_RECORD_TYPE_EMG = 1;
constexpr uint8_t BLE_RECORD_TYPE_IMU = 2;
constexpr size_t BLE_PACKET_HEADER_BYTES = 8;
constexpr size_t EMG_SAMPLE_BYTES = 24;
constexpr size_t IMU_SAMPLE_BYTES = 14;
constexpr size_t EMG_PACKET_METADATA_BYTES = 15;
constexpr size_t IMU_PACKET_METADATA_BYTES = 12;
constexpr size_t BLE_MAX_ATT_PAYLOAD_BYTES = 244;
constexpr uint16_t BLE_REQUESTED_MTU = 247;
constexpr uint32_t EMG_BATCH_WAIT_US = 20000;
constexpr uint32_t IMU_BATCH_WAIT_US = 160000;

SPISettings registerSpi(100000, MSBFIRST, SPI_MODE1);
SPISettings dataSpi(4000000, MSBFIRST, SPI_MODE1);
Lsm6dsoxSpi imu(SPI, PIN_CS_IMU);

struct Sample {
  uint32_t sequence;
  uint32_t timestampUs;
  uint32_t status;
  int32_t channels[ADS_CHANNEL_COUNT];
  uint32_t dropped;
};

struct ImuSample {
  uint32_t sequence;
  uint32_t timestampUs;
  int16_t temperature;
  int16_t gyro[3];
  int16_t acceleration[3];
  uint32_t dropped;
};

QueueHandle_t sampleQueue = nullptr;
QueueHandle_t imuQueue = nullptr;
TaskHandle_t acquisitionTaskHandle = nullptr;

uint32_t nextSequence = 0;
uint32_t droppedSamples = 0;
uint32_t nextImuSequence = 0;
uint32_t droppedImuSamples = 0;
volatile uint32_t acquisitionHeartbeat = 0;
uint32_t lastHeartbeat = 0;
uint32_t lastHeartbeatCheckMs = 0;
bool streaming = false;

BLEServer *bleServer = nullptr;
BLECharacteristic *bleDataCharacteristic = nullptr;
volatile bool bleConnected = false;
volatile bool bleSubscribed = false;
volatile bool bleFlushRequested = false;
volatile uint16_t bleConnectionHandle = BLE_HS_CONN_HANDLE_NONE;
volatile uint16_t blePeerMtu = 23;
uint16_t nextBlePacketSequence = 0;
bool bleMtuErrorReported = false;

class EmgBleServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server, ble_gap_conn_desc *description) override {
    bleConnected = true;
    bleSubscribed = false;
    bleFlushRequested = true;
    if (description != nullptr) {
      bleConnectionHandle = description->conn_handle;
      blePeerMtu = server->getPeerMTU(description->conn_handle);
      server->requestConnParams(description->conn_handle, 6, 12, 0, 200);
    }
    Serial.println("# INFO,BLE_CONNECTED");
  }

  void onDisconnect(BLEServer *server, ble_gap_conn_desc *description) override {
    (void)server;
    (void)description;
    bleConnected = false;
    bleSubscribed = false;
    bleConnectionHandle = BLE_HS_CONN_HANDLE_NONE;
    blePeerMtu = 23;
    Serial.println("# INFO,BLE_DISCONNECTED");
    BLEDevice::startAdvertising();
    const bool advertisingActive =
        BLEDevice::getAdvertising()->isAdvertising();
    Serial.printf("# INFO,BLE_ADVERTISING_RESTART,active=%u\r\n",
                  static_cast<unsigned>(advertisingActive));
  }

  void onMtuChanged(BLEServer *server, ble_gap_conn_desc *description,
                    uint16_t mtu) override {
    (void)server;
    (void)description;
    blePeerMtu = mtu;
    Serial.printf("# INFO,BLE_MTU,%u\r\n", static_cast<unsigned>(mtu));
  }
};

class EmgBleDataCallbacks : public BLECharacteristicCallbacks {
  void onSubscribe(BLECharacteristic *characteristic,
                   ble_gap_conn_desc *description,
                   uint16_t subscriptionValue) override {
    (void)characteristic;
    bleSubscribed = (subscriptionValue & 0x0001U) != 0;
    if (description != nullptr) {
      bleConnectionHandle = description->conn_handle;
    }
    if (bleSubscribed) {
      bleFlushRequested = true;
      Serial.println("# INFO,BLE_NOTIFY_ENABLED");
    } else {
      Serial.println("# INFO,BLE_NOTIFY_DISABLED");
    }
  }
};

EmgBleServerCallbacks bleServerCallbacks;
EmgBleDataCallbacks bleDataCallbacks;

int32_t decodeSigned24(const uint8_t *bytes) {
  int32_t value = (static_cast<int32_t>(bytes[0]) << 16) |
                  (static_cast<int32_t>(bytes[1]) << 8) |
                  static_cast<int32_t>(bytes[2]);

  if ((value & 0x00800000L) != 0) {
    value |= 0xFF000000L;
  }
  return value;
}

uint32_t decodeStatus24(const uint8_t *bytes) {
  return (static_cast<uint32_t>(bytes[0]) << 16) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         static_cast<uint32_t>(bytes[2]);
}

bool runDecoderSelfTest() {
  const uint8_t zero[3] = {0x00, 0x00, 0x00};
  const uint8_t maxPositive[3] = {0x7F, 0xFF, 0xFF};
  const uint8_t minNegative[3] = {0x80, 0x00, 0x00};
  const uint8_t minusOne[3] = {0xFF, 0xFF, 0xFF};
  const uint8_t status[3] = {0xC0, 0x00, 0x00};

  return decodeSigned24(zero) == 0 &&
         decodeSigned24(maxPositive) == 8388607 &&
         decodeSigned24(minNegative) == -8388608 &&
         decodeSigned24(minusOne) == -1 &&
         decodeStatus24(status) == 12582912UL;
}

void ARDUINO_ISR_ATTR onDrdyFalling() {
  if (acquisitionTaskHandle == nullptr) {
    return;
  }

  BaseType_t higherPriorityTaskWoken = pdFALSE;
  vTaskNotifyGiveFromISR(acquisitionTaskHandle, &higherPriorityTaskWoken);
  if (higherPriorityTaskWoken == pdTRUE) {
    portYIELD_FROM_ISR();
  }
}

void adsSelect(uint16_t setupUs = 20) {
  delayMicroseconds(setupUs);
  digitalWrite(PIN_CS_ADS, LOW);
  delayMicroseconds(setupUs);
}

void adsDeselect(uint16_t holdUs = 20, uint16_t recoveryUs = 40) {
  delayMicroseconds(holdUs);
  digitalWrite(PIN_CS_ADS, HIGH);
  delayMicroseconds(recoveryUs);
}

void adsCommand(uint8_t command) {
  SPI.beginTransaction(registerSpi);
  adsSelect();
  SPI.transfer(command);
  adsDeselect();
  SPI.endTransaction();
  delayMicroseconds(80);
}

bool waitForDrdyLow(uint32_t timeoutMs) {
  const uint32_t startMs = millis();
  while (digitalRead(PIN_DRDY) == HIGH) {
    if (millis() - startMs >= timeoutMs) {
      return false;
    }
    delay(1);
  }
  return true;
}

void resetAdsHardware() {
  digitalWrite(PIN_CS_ADS, HIGH);
  digitalWrite(PIN_PWDN, HIGH);
  digitalWrite(PIN_RESET, LOW);
  delay(20);
  digitalWrite(PIN_RESET, HIGH);
  delay(500);
}

void stopContinuousModeRobustly() {
  for (uint8_t attempt = 0; attempt < 10; ++attempt) {
    waitForDrdyLow(20);
    delayMicroseconds(300);
    adsCommand(CMD_SDATAC);
    delay(3);
  }
}

void readRegisters(uint8_t startAddress, uint8_t *buffer, uint8_t count) {
  SPI.beginTransaction(registerSpi);
  adsSelect();
  SPI.transfer(CMD_RREG | (startAddress & 0x1F));
  delayMicroseconds(80);
  SPI.transfer((count - 1) & 0x1F);
  delayMicroseconds(80);

  for (uint8_t i = 0; i < count; ++i) {
    buffer[i] = SPI.transfer(0x00);
  }

  adsDeselect();
  SPI.endTransaction();
}

void writeRegisters(uint8_t startAddress, const uint8_t *data, uint8_t count) {
  SPI.beginTransaction(registerSpi);
  adsSelect();
  SPI.transfer(CMD_WREG | (startAddress & 0x1F));
  delayMicroseconds(80);
  SPI.transfer((count - 1) & 0x1F);
  delayMicroseconds(80);

  for (uint8_t i = 0; i < count; ++i) {
    SPI.transfer(data[i]);
    delayMicroseconds(80);
  }

  adsDeselect();
  SPI.endTransaction();
  delayMicroseconds(80);
}

bool verifyRegister(uint8_t address, uint8_t expected, uint8_t actual) {
  if (actual == expected) {
    return true;
  }

  Serial.printf("# ERROR,CONFIG_VERIFY,address=0x%02X,expected=0x%02X,actual=0x%02X\r\n",
                address, expected, actual);
  return false;
}

bool initializeAds1298() {
  resetAdsHardware();
  stopContinuousModeRobustly();

  uint8_t id = 0;
  readRegisters(REG_ID, &id, 1);
  if (id != ADS1298_ID) {
    Serial.printf("# ERROR,ADS_ID,expected=0x%02X,actual=0x%02X\r\n", ADS1298_ID, id);
    return false;
  }

  const uint8_t configs[3] = {
      CONFIG1_500_SPS_HR,
      CONFIG2_TEST_OFF,
      CONFIG3_RLD_ON_INTERNAL_REF,
  };
  const uint8_t channels[ADS_CHANNEL_COUNT] = {
      CHANNEL_NORMAL_GAIN6, CHANNEL_NORMAL_GAIN6,
      CHANNEL_NORMAL_GAIN6, CHANNEL_NORMAL_GAIN6,
      CHANNEL_NORMAL_GAIN6, CHANNEL_NORMAL_GAIN6,
      CHANNEL_NORMAL_GAIN6, CHANNEL_NORMAL_GAIN6,
  };
  const uint8_t rldSense[2] = {
      RLD_SENSP_CH1,
      RLD_SENSN_CH1,
  };

  writeRegisters(REG_CONFIG1, configs, sizeof(configs));
  writeRegisters(REG_CH1SET, channels, sizeof(channels));
  writeRegisters(REG_RLD_SENSP, rldSense, sizeof(rldSense));
  delay(200);

  uint8_t registers[15] = {0};
  readRegisters(REG_ID, registers, sizeof(registers));

  bool valid = verifyRegister(REG_ID, ADS1298_ID, registers[REG_ID]);
  valid &= verifyRegister(REG_CONFIG1, configs[0], registers[REG_CONFIG1]);
  valid &= verifyRegister(REG_CONFIG2, configs[1], registers[REG_CONFIG2]);
  valid &= verifyRegister(REG_CONFIG3, configs[2], registers[REG_CONFIG3]);
  for (uint8_t channel = 0; channel < ADS_CHANNEL_COUNT; ++channel) {
    const uint8_t address = REG_CH1SET + channel;
    valid &= verifyRegister(address, CHANNEL_NORMAL_GAIN6, registers[address]);
  }
  valid &= verifyRegister(REG_RLD_SENSP, rldSense[0], registers[REG_RLD_SENSP]);
  valid &= verifyRegister(REG_RLD_SENSN, rldSense[1], registers[REG_RLD_SENSN]);
  if (!valid) {
    return false;
  }

  adsCommand(CMD_START);
  delay(10);
  adsCommand(CMD_RDATAC);
  return true;
}

void readDataFrame(uint8_t *frame) {
  SPI.beginTransaction(dataSpi);
  adsSelect(2);
  for (uint8_t i = 0; i < ADS_FRAME_BYTES; ++i) {
    frame[i] = SPI.transfer(0x00);
  }
  adsDeselect(2, 2);
  SPI.endTransaction();
}

void putU16(uint8_t *buffer, size_t &offset, uint16_t value) {
  buffer[offset++] = static_cast<uint8_t>(value & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 8) & 0xFFU);
}

void putI16(uint8_t *buffer, size_t &offset, int16_t value) {
  putU16(buffer, offset, static_cast<uint16_t>(value));
}

void putU24(uint8_t *buffer, size_t &offset, uint32_t value) {
  buffer[offset++] = static_cast<uint8_t>(value & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 8) & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 16) & 0xFFU);
}

void putI24(uint8_t *buffer, size_t &offset, int32_t value) {
  putU24(buffer, offset, static_cast<uint32_t>(value) & 0x00FFFFFFUL);
}

void putU32(uint8_t *buffer, size_t &offset, uint32_t value) {
  buffer[offset++] = static_cast<uint8_t>(value & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 8) & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 16) & 0xFFU);
  buffer[offset++] = static_cast<uint8_t>((value >> 24) & 0xFFU);
}

bool runBleEncoderSelfTest() {
  uint8_t bytes[16] = {0};
  size_t offset = 0;
  putU16(bytes, offset, 0x1234U);
  putU24(bytes, offset, 0x00C0ABCDUL);
  putI24(bytes, offset, -1);
  putU32(bytes, offset, 0x89ABCDEFUL);
  putI16(bytes, offset, -32768);

  const uint8_t expected[] = {
      0x34, 0x12, 0xCD, 0xAB, 0xC0, 0xFF, 0xFF,
      0xFF, 0xEF, 0xCD, 0xAB, 0x89, 0x00, 0x80,
  };
  if (offset != sizeof(expected)) {
    return false;
  }
  for (size_t index = 0; index < sizeof(expected); ++index) {
    if (bytes[index] != expected[index]) {
      return false;
    }
  }
  return true;
}

void putBleHeader(uint8_t *buffer, uint8_t recordType, uint8_t recordCount,
                  uint8_t recordBytes, uint16_t packetSequence) {
  size_t offset = 0;
  buffer[offset++] = BLE_PACKET_MAGIC_0;
  buffer[offset++] = BLE_PACKET_MAGIC_1;
  buffer[offset++] = BLE_PROTOCOL_VERSION;
  buffer[offset++] = recordType;
  buffer[offset++] = recordCount;
  buffer[offset++] = recordBytes;
  putU16(buffer, offset, packetSequence);
}

size_t currentBlePayloadLimit() {
  if (!bleConnected || bleServer == nullptr ||
      bleConnectionHandle == BLE_HS_CONN_HANDLE_NONE) {
    return 0;
  }

  const uint16_t reportedMtu = bleServer->getPeerMTU(bleConnectionHandle);
  if (reportedMtu >= 23) {
    blePeerMtu = reportedMtu;
  }
  if (blePeerMtu <= 3) {
    return 0;
  }
  const size_t payloadBytes = static_cast<size_t>(blePeerMtu - 3);
  return payloadBytes < BLE_MAX_ATT_PAYLOAD_BYTES
             ? payloadBytes
             : BLE_MAX_ATT_PAYLOAD_BYTES;
}

bool notifyBlePacket(uint8_t *packet, size_t packetBytes) {
  if (!bleConnected || !bleSubscribed || bleDataCharacteristic == nullptr) {
    return false;
  }
  bleDataCharacteristic->setValue(packet, packetBytes);
  bleDataCharacteristic->notify();
  ++nextBlePacketSequence;
  return true;
}

bool drainOneBleEmgPacket() {
  Sample oldest = {};
  if (xQueuePeek(sampleQueue, &oldest, 0) != pdTRUE) {
    return false;
  }

  const size_t payloadLimit = currentBlePayloadLimit();
  const size_t minimumPacketBytes = BLE_PACKET_HEADER_BYTES +
                                    EMG_PACKET_METADATA_BYTES +
                                    EMG_SAMPLE_BYTES;
  if (payloadLimit < minimumPacketBytes) {
    if (!bleMtuErrorReported) {
      Serial.printf("# ERROR,BLE_MTU_TOO_SMALL,mtu=%u,required=%u\r\n",
                    static_cast<unsigned>(blePeerMtu),
                    static_cast<unsigned>(minimumPacketBytes + 3));
      bleMtuErrorReported = true;
    }
    return false;
  }
  bleMtuErrorReported = false;

  const size_t maximumRecords =
      (payloadLimit - BLE_PACKET_HEADER_BYTES -
       EMG_PACKET_METADATA_BYTES) /
      EMG_SAMPLE_BYTES;
  const size_t availableRecords = uxQueueMessagesWaiting(sampleQueue);
  if (availableRecords < maximumRecords &&
      static_cast<uint32_t>(micros() - oldest.timestampUs) <
          EMG_BATCH_WAIT_US) {
    return false;
  }
  const size_t requestedRecords =
      availableRecords < maximumRecords ? availableRecords : maximumRecords;

  uint8_t packet[BLE_MAX_ATT_PAYLOAD_BYTES] = {0};
  size_t offset = BLE_PACKET_HEADER_BYTES;
  putU32(packet, offset, oldest.sequence);
  putU32(packet, offset, oldest.timestampUs);
  putU24(packet, offset, oldest.status);
  putU32(packet, offset, oldest.dropped);
  uint8_t packedRecords = 0;
  for (size_t index = 0; index < requestedRecords; ++index) {
    Sample candidate = {};
    if (xQueuePeek(sampleQueue, &candidate, 0) != pdTRUE ||
        candidate.sequence != oldest.sequence + packedRecords) {
      break;
    }
    Sample sample = {};
    if (xQueueReceive(sampleQueue, &sample, 0) != pdTRUE) {
      break;
    }
    for (uint8_t channel = 0; channel < ADS_CHANNEL_COUNT; ++channel) {
      putI24(packet, offset, sample.channels[channel]);
    }
    ++packedRecords;
  }
  if (packedRecords == 0) {
    return false;
  }

  putBleHeader(packet, BLE_RECORD_TYPE_EMG, packedRecords,
               static_cast<uint8_t>(EMG_SAMPLE_BYTES),
               nextBlePacketSequence);
  return notifyBlePacket(packet, offset);
}

bool drainOneBleImuPacket() {
  ImuSample oldest = {};
  if (xQueuePeek(imuQueue, &oldest, 0) != pdTRUE) {
    return false;
  }

  const size_t payloadLimit = currentBlePayloadLimit();
  const size_t minimumPacketBytes = BLE_PACKET_HEADER_BYTES +
                                    IMU_PACKET_METADATA_BYTES +
                                    IMU_SAMPLE_BYTES;
  if (payloadLimit < minimumPacketBytes) {
    return false;
  }
  const size_t maximumRecords =
      (payloadLimit - BLE_PACKET_HEADER_BYTES -
       IMU_PACKET_METADATA_BYTES) /
      IMU_SAMPLE_BYTES;
  const size_t availableRecords = uxQueueMessagesWaiting(imuQueue);
  if (availableRecords < maximumRecords &&
      static_cast<uint32_t>(micros() - oldest.timestampUs) <
          IMU_BATCH_WAIT_US) {
    return false;
  }
  const size_t requestedRecords =
      availableRecords < maximumRecords ? availableRecords : maximumRecords;

  uint8_t packet[BLE_MAX_ATT_PAYLOAD_BYTES] = {0};
  size_t offset = BLE_PACKET_HEADER_BYTES;
  putU32(packet, offset, oldest.sequence);
  putU32(packet, offset, oldest.timestampUs);
  putU32(packet, offset, oldest.dropped);
  uint8_t packedRecords = 0;
  for (size_t index = 0; index < requestedRecords; ++index) {
    ImuSample candidate = {};
    if (xQueuePeek(imuQueue, &candidate, 0) != pdTRUE ||
        candidate.sequence != oldest.sequence + packedRecords) {
      break;
    }
    ImuSample sample = {};
    if (xQueueReceive(imuQueue, &sample, 0) != pdTRUE) {
      break;
    }
    for (uint8_t axis = 0; axis < 3; ++axis) {
      putI16(packet, offset, sample.acceleration[axis]);
    }
    for (uint8_t axis = 0; axis < 3; ++axis) {
      putI16(packet, offset, sample.gyro[axis]);
    }
    putI16(packet, offset, sample.temperature);
    ++packedRecords;
  }
  if (packedRecords == 0) {
    return false;
  }

  putBleHeader(packet, BLE_RECORD_TYPE_IMU, packedRecords,
               static_cast<uint8_t>(IMU_SAMPLE_BYTES),
               nextBlePacketSequence);
  return notifyBlePacket(packet, offset);
}

void initializeBle() {
  BLEDevice::init(BLE_DEVICE_NAME);
  BLEDevice::setMTU(247);
  BLEDevice::setPower(ESP_PWR_LVL_N6);

  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(&bleServerCallbacks);

  BLEService *service = bleServer->createService(BLE_SERVICE_UUID);
  bleDataCharacteristic = service->createCharacteristic(
      BLE_DATA_CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  bleDataCharacteristic->setCallbacks(&bleDataCallbacks);
  bleDataCharacteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising *advertising = bleServer->getAdvertising();
  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMaxPreferred(0x12);
  const String bleAddress = BLEDevice::getAddress().toString();
  const bool advertisingStarted = advertising->start();
  const bool advertisingActive = advertising->isAdvertising();
  Serial.printf(
      "# INFO,BLE_ADVERTISING,name=%s,address=%s,started=%u,active=%u\r\n",
      BLE_DEVICE_NAME, bleAddress.c_str(),
      static_cast<unsigned>(advertisingStarted),
      static_cast<unsigned>(advertisingActive));
  if (!advertisingStarted || !advertisingActive) {
    Serial.println("# ERROR,BLE_ADVERTISING_START");
  }
}

void acquisitionTask(void *parameter) {
  (void)parameter;

  while (true) {
    const uint32_t events = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (events > 1) {
      const uint32_t missed = events - 1;
      droppedSamples += missed;
      nextSequence += missed;
    }

    uint8_t frame[ADS_FRAME_BYTES] = {0};
    readDataFrame(frame);

    Sample sample = {};
    sample.sequence = nextSequence++;
    sample.timestampUs = micros();
    sample.status = decodeStatus24(frame);
    for (uint8_t channel = 0; channel < ADS_CHANNEL_COUNT; ++channel) {
      sample.channels[channel] = decodeSigned24(&frame[3 + channel * 3]);
    }
    sample.dropped = droppedSamples;

    if (xQueueSend(sampleQueue, &sample, 0) != pdTRUE) {
      ++droppedSamples;
    }

    Lsm6dsoxRawSample rawImu = {};
    if (imu.readIfReady(rawImu)) {
      ImuSample imuSample = {};
      imuSample.sequence = nextImuSequence++;
      imuSample.timestampUs = micros();
      imuSample.temperature = rawImu.temperature;
      for (uint8_t axis = 0; axis < 3; ++axis) {
        imuSample.gyro[axis] = rawImu.gyro[axis];
        imuSample.acceleration[axis] = rawImu.acceleration[axis];
      }
      imuSample.dropped = droppedImuSamples;
      if (xQueueSend(imuQueue, &imuSample, 0) != pdTRUE) {
        ++droppedImuSamples;
      }
    }
    ++acquisitionHeartbeat;
  }
}

void setup() {
  pinMode(PIN_CS_ADS, OUTPUT);
  digitalWrite(PIN_CS_ADS, HIGH);

  pinMode(PIN_CS_IMU, OUTPUT);
  digitalWrite(PIN_CS_IMU, HIGH);

  pinMode(PIN_PWDN, OUTPUT);
  digitalWrite(PIN_PWDN, HIGH);

  pinMode(PIN_RESET, OUTPUT);
  digitalWrite(PIN_RESET, HIGH);

  pinMode(PIN_DRDY, INPUT_PULLUP);
  SPI.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS_ADS);

  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);

  if (!runDecoderSelfTest()) {
    Serial.println("# ERROR,SELFTEST,decoder");
    return;
  }
  if (!runBleEncoderSelfTest()) {
    Serial.println("# ERROR,SELFTEST,ble_encoder");
    return;
  }

  uint8_t imuDeviceId = 0;
  Serial.println("# INFO,LSM6DSOX_INITIALIZING");
  if (!imu.begin(imuDeviceId)) {
    Serial.printf("# ERROR,LSM6DSOX_INIT,expected=0x6C,actual=0x%02X\r\n",
                  imuDeviceId);
    return;
  }
  Serial.println("# INFO,ADS1298_INITIALIZING");

  if (!initializeAds1298()) {
    return;
  }

  Serial.println("# SELFTEST,OK");
  Serial.println("# ADS1298,transport=BLE,rld=on,rld_sensp=0x01,rld_sensn=0x01,id=0x92,sps=500,gain=6,vref_mV=2400");
  Serial.println("# LSM6DSOX,transport=BLE,model=LSM6DSOX,id=0x6C,odr_hz=104,accel_fs_g=4,gyro_fs_dps=500");
  Serial.println("# BLE_PROTOCOL,version=2,header_bytes=8,emg_metadata_bytes=15,emg_sample_bytes=24,imu_metadata_bytes=12,imu_sample_bytes=14,byte_order=little_endian");

  sampleQueue = xQueueCreate(SAMPLE_QUEUE_CAPACITY, sizeof(Sample));
  if (sampleQueue == nullptr) {
    Serial.println("# ERROR,QUEUE_CREATE");
    return;
  }

  imuQueue = xQueueCreate(IMU_QUEUE_CAPACITY, sizeof(ImuSample));
  if (imuQueue == nullptr) {
    Serial.println("# ERROR,IMU_QUEUE_CREATE");
    return;
  }

  initializeBle();

  if (xTaskCreate(acquisitionTask, "ads1298_acq", 4096, nullptr, 5,
                  &acquisitionTaskHandle) != pdPASS) {
    Serial.println("# ERROR,TASK_CREATE");
    return;
  }
  attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdyFalling, FALLING);

  lastHeartbeat = acquisitionHeartbeat;
  lastHeartbeatCheckMs = millis();
  streaming = true;
}

void loop() {
  if (!streaming) {
    delay(100);
    return;
  }

  if (bleFlushRequested) {
    xQueueReset(sampleQueue);
    xQueueReset(imuQueue);
    bleFlushRequested = false;
  }

  bool packetSent = false;
  if (bleConnected && bleSubscribed) {
    packetSent = drainOneBleEmgPacket();
    packetSent = drainOneBleImuPacket() || packetSent;
  }
  if (!packetSent) {
    delay(1);
  }

  const uint32_t nowMs = millis();
  if (nowMs - lastHeartbeatCheckMs >= 1000) {
    const uint32_t currentHeartbeat = acquisitionHeartbeat;
    if (currentHeartbeat == lastHeartbeat) {
      Serial.println("# ERROR,DRDY_TIMEOUT");
    }
    lastHeartbeat = currentHeartbeat;
    lastHeartbeatCheckMs = nowMs;
  }
}
