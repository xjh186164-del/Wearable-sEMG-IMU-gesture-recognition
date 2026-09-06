#pragma once

#include <Arduino.h>
#include <SPI.h>

struct Lsm6dsoxRawSample {
  int16_t temperature;
  int16_t gyro[3];
  int16_t acceleration[3];
};

class Lsm6dsoxSpi {
 public:
  Lsm6dsoxSpi(SPIClass &spi, uint8_t chipSelectPin);

  bool begin(uint8_t &deviceId);
  bool readIfReady(Lsm6dsoxRawSample &sample);
  uint8_t readRegister(uint8_t address);

 private:
  void writeRegister(uint8_t address, uint8_t value);
  void readRegisters(uint8_t address, uint8_t *data, size_t length);

  SPIClass &spi_;
  uint8_t chipSelectPin_;
  SPISettings settings_;
};
