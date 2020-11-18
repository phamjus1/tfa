import socket
import random
import face_tracker_v3 as ft3

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
ftracker = ft3.FTC()

while(1):
    sd = ftracker.execute()
    print(sd)
    sendData = sd.encode()
    conn, addr = s.accept()
    print('Connected by', addr)
    #print(sendData)
    #print(sendData.encode().__sizeof__())
    conn.sendall(sendData)

s.close()
