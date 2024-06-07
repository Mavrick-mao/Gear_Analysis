import socket

# サーバーの設定
HOST = '0.0.0.0'  # サーバーをすべてのネットワークインターフェースで受け付ける
PORT = 8080  # 使用するポート番号を指定

# ソケットを作成し、指定したホストとポートで接続待ち状態にする
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))  # ホストとポートをバインド
    server_socket.listen()  # 接続待ち状態にする
    print(f"サーバーがポート {PORT} で起動しました")

    # クライアントからの接続を待機
    conn, addr = server_socket.accept()
    with conn:
        print('クライアントが接続しました:', addr)
        while True:
            data = conn.recv(1000000)  # クライアントからデータを受信 (最大1024バイト)
            if not data:
                break
            print("Arduinoからのデータ:", data.decode('utf-8'))  # データをUTF-8でデコードして表示e.encode())