import socket
import random

HOST = '127.0.0.1'
PORT = 50014

class Data():

    def emit_data(self):
        rnums = []
        for i in range(9):
            rnums.append(random.randint(0,9))
        s = "{}.{}{};{}.{}{};{}.{}{}"
        return  s.format(*rnums)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()

d = Data()

while(1):
    conn, addr = s.accept()
    print('Connected by', addr)
    sendData = d.emit_data()
    #print(sendData)
    #print(sendData.encode().__sizeof__())
    conn.sendall(sendData.encode())

s.close()
