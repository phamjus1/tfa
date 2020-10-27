using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System;

public class TrackClient : MonoBehaviour
{
    public int port;
    public string host;
    public string responseData;
    private TcpClient client;
    private NetworkStream stream;
    private byte[] data;

    // Start is called before the first frame update
    void Start()
    {
        port = 50014;
        host = "127.0.0.1";
        data = new byte[256];
        
    }

    // Update is called once per frame
    void Update()
    {
        try {
            client = new TcpClient(host, port);
            stream = client.GetStream();

            // Read and Process header data in packet
            System.Int32 bytes = stream.Read(data, 0, 47);
            responseData = String.Empty;
            responseData = System.Text.Encoding.ASCII.GetString(data, 0, 47);
            Debug.Log("Received: " + responseData);
            stream.Close();
            client.Close();
        } catch (ArgumentNullException e) {
        } catch (SocketException e) {
        }
    }
}
