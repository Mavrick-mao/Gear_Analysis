int b_pin = 7;   // デジタルデータ出力用ピン番号
int state = 0;   // ピンより取得したデータ格納用

void setup() {
  Serial.begin(9600);  
  pinMode(b_pin, INPUT);     // ボタンスイッチ用に入力に設定
}

void loop() {
  state = digitalRead(b_pin);  // ピンよりデータ取得
  Serial.println(state);       // シリアルモニタに出力
  delay(450);
}