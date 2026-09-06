#include "lsm6dsox_spi.h"

namespace {
constexpr uint8_t LSM6DSOX_ID = 0x6C;

constexpr uint8_t REG_WHO_AM_I = 0x0F;
constexpr uint8_t REG_CTRL1_XL = 0x10;
constexpr uint8_t REG_CTRL2_G = 0x11;
constexpr uint8_t REG_CTRL3_C = 0x12;
constexpr uint8_t REG_STATUS = 0x1E;
constexpr uint8_t REG_OUT_TEMP_L = 0x20;

constexpr uint8_t CTRL1_XL_104HZ_4G = 0x48;
constexpr uint8_t CTRL2_G_104HZ_500DPS = 0x44;
constexpr uint8_t CTRL3_C_BDU_IF_INC = 0x44;
constexpr uint8_t CTRL3_C_SW_RESET = 0x01;

constexpr uint8_t STATUS_XLDA = 0x01;
constexpr uint8_t STATUS_GDA = 0x02;
constexpr uint8_t SPI_READ = 0x80;
constexpr uint32_t RESET_TIMEOUT_MS = 100;

int16_t decodeSigned16(const uint8_t *bytes) {
  const uint16_t value = static_cast<uint16_t>(bytes[0]) |
                         (static_cast<uint16_t>(bytes[1]) << 8);
  return static_cast<int16_t>(value);
}
}  // namespace

Lsm6dsoxSpi::Lsm6dsoxSpi(SPIClass &spi, uint8_t chipSelectPin)
    : spi_(spi),
      chipSelectPin_(chipSelectPin),
      settings_(1000000, MSBFIRST, SPI_MODE3) {}

bool Lsm6dsoxSpi::begin(uint8_t &deviceId) {
  pinMode(chipSelectPin_, OUTPUT);
  digitalWrite(chipSelectPin_, HIGH);
  delay(10);

  deviceId = readRegister(REG_WHO_AM_I);
  if (deviceId != LSM6DSOX_ID) {
    return false;
  }

  writeRegister(REG_CTRL3_C, CTRL3_C_SW_RESET);
  const uint32_t resetStartMs = millis();
  while ((readRegister(REG_CTRL3_C) & CTRL3_C_SW_RESET) != 0) {
    if (millis() - resetStartMs >= RESET_TIMEOUT_MS) {
      return false;
    }
    delay(1);
  }

  writeRegister(REG_CTRL3_C, CTRL3_C_BDU_IF_INC);
  writeRegister(REG_CTRL1_XL, CTRL1_XL_104HZ_4G);
  writeRegister(REG_CTRL2_G, CTRL2_G_104HZ_500DPS);
  delay(20);

  return readRegister(REG_CTRL3_C) == CTRL3_C_BDU_IF_INC &&
         readRegister(REG_CTRL1_XL) == CTRL1_XL_104HZ_4G &&
         readRegister(REG_CTRL2_G) == CTRL2_G_104HZ_500DPS;
}

bool Lsm6dsoxSpi::readIfReady(Lsm6dsoxRawSample &sample) {
  const uint8_t status = readRegister(REG_STATUS);
  if ((status & (STATUS_XLDA | STATUS_GDA)) != (STATUS_XLDA | STATUS_GDA)) {
    return false;
  }

  uint8_t bytes[14] = {0};
  readRegisters(REG_OUT_TEMP_L, bytes, sizeof(bytes));
  sample.temperature = decodeSigned16(&bytes[0]);
  for (uint8_t axis = 0; axis < 3; ++axis) {
    sample.gyro[axis] = decodeSigned16(&bytes[2 + axis * 2]);
    sample.acceleration[axis] = decodeSigned16(&bytes[8 + axis * 2]);
  }
  return true;
}

uint8_t Lsm6dsoxSpi::readRegister(uint8_t address) {
  uint8_t value = 0;
  readRegisters(address, &value, 1);
  return value;
}

void Lsm6dsoxSpi::writeRegister(uint8_t address, uint8_t value) {
  spi_.beginTransaction(settings_);
  digitalWrite(chipSelectPin_, LOW);
  delayMicroseconds(1);
  spi_.transfer(address & ~SPI_READ);
  spi_.transfer(value);
  delayMicroseconds(1);
  digitalWrite(chipSelectPin_, HIGH);
  spi_.endTransaction();
}

void Lsm6dsoxSpi::readRegisters(uint8_t address, uint8_t *data, size_t length) {
  spi_.beginTransaction(settings_);
  digitalWrite(chipSelectPin_, LOW);
  delayMicroseconds(1);
  spi_.transfer(address | SPI_READ);
  for (size_t index = 0; index < length; ++index) {
    data[index] = spi_.transfer(0x00);
  }
  delayMicroseconds(1);
  digitalWrite(chipSelectPin_, HIGH);
  spi_.endTransaction();
}
