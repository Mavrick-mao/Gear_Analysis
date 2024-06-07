#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>
#include <iostream>
#include <tuple>

const char* ssid = "Buffalo-G-46D8";
const char* password = "wg4yyx4sv3dvu";
const char* serverName = "https://script.google.com/macros/s/AKfycbxGa5frcYFlSetmA5UjjxEIraQGsCi_81FVYRFHhcXyTKd7Rq8eNCT0qegr1GO56ADa/exec";
const char* ID = "研究室";

unsigned long startTime;
unsigned long endTime;
unsigned long duringTime;

const int ModePin = 12;
const int ButtonPin = 14;

const int voltage1Pin = 34;
const int voltage2Pin = 35;

const int greenled = 15;
const int redled = 17;

String dataLog1;
String dataLog2;

void blink() {
  for (int i = 0; i < 10; i++) {
    digitalWrite(greenled, HIGH);
    digitalWrite(redled, HIGH);
    delay(50); // 0.5秒待つ
    digitalWrite(greenled, LOW);
    digitalWrite(redled, LOW);
    delay(500); // 0.5秒待つ
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(ModePin, INPUT_PULLUP);
  pinMode(ButtonPin, INPUT_PULLUP);
  pinMode(voltage1Pin, INPUT);
  pinMode(voltage2Pin, INPUT);
  pinMode(greenled, OUTPUT);
  pinMode(redled, OUTPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi..");
  }
  Serial.println("Connected to WiFi");
  blink();
}

void postToSpreadsheet(String IMEI, long duringTime, String voltage1, String voltage2, String status) {
  digitalWrite(redled, HIGH);
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String httpRequestData = "{\"id\":\"" + IMEI + "\",\"duringtime\":\"" + duringTime + "\",\"voltage_1\":\"" + voltage1 + "\",\"voltage_2\":\"" + voltage2 + "\",\"status\":\"" + status + "\"}";
    int contentLength = httpRequestData.length();
    http.addHeader("Content-Length", String(contentLength));

    // Serial.println(httpRequestData);

    int httpResponseCode = http.POST(httpRequestData);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.println("Error on sending POST: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
  digitalWrite(redled, LOW);
}

void loop() {
  if (digitalRead(ButtonPin) == HIGH) {
    int mic1_data;
    int mic2_data;
    int count = 1;
    String dataLog1 = "";
    String dataLog2 = "";
    startTime = millis();
    digitalWrite(greenled, HIGH);

    if (digitalRead(ModePin) == HIGH) {
      Serial.println("Mode: Test ");
      Serial.println("Now Recording >>> ");
      while (digitalRead(ButtonPin) == HIGH && count < 2000) {
        mic1_data = analogRead(voltage1Pin);
        mic2_data = analogRead(voltage2Pin);
        float voltage1 = mic1_data * 5.0f / 4095.0f;
        float voltage2 = mic2_data * 5.0f / 4095.0f;
        dataLog1 += String(voltage1, 4) + ",";
        dataLog2 += String(voltage2, 4) + ",";
        count++;
        delay(1);
        // Serial.println(count);
      }
    } else {
      Serial.println("Mode: Nan ");
      Serial.println("Now Recording >>> ");
      while (digitalRead(ButtonPin) == HIGH) {
        mic1_data = analogRead(voltage1Pin);
        mic2_data = analogRead(voltage2Pin);
        float voltage1 = mic1_data * 5.0f / 4095.0f;
        float voltage2 = mic2_data * 5.0f / 4095.0f;
        dataLog1 += String(voltage1, 4) + ",";
        dataLog2 += String(voltage2, 4) + ",";
        delay(1);
      }
    }
    digitalWrite(greenled, LOW);
    Serial.println("Recording End>>> ");

    endTime = millis();
    duringTime = endTime - startTime;

    Serial.println(duringTime);
    Serial.println(dataLog1.length());

    if(dataLog1.length() > 0 || dataLog2.length() > 0) {
      Serial.println("Data Uploading");
      if (digitalRead(ButtonPin) == HIGH) {
        postToSpreadsheet(ID, duringTime, dataLog1, dataLog2, "True");
      } else {
        postToSpreadsheet(ID, duringTime, dataLog1, dataLog2, "False");
      }
    }
  }
}
