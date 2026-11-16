#define SENSOR_PIN A0

const unsigned long SAMPLE_INTERVAL_US = 4000; // 250 Hz
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(10);   // 10-bit = 0 to 1023
  lastSampleTime = micros();
}

void loop() {
  unsigned long currentTime = micros();

  if ((unsigned long)(currentTime - lastSampleTime) >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    int eogValue = analogRead(SENSOR_PIN);
    Serial.println(eogValue);
  }
}