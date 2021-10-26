# Network

### 1.1 A day in the life of an application

##### WWW (world wide wed)

Throught `http`, two servers can **send request** and **receive response** from another servers

##### Bittorrent

The most easiest way for two computers to transfer information is by streaming, which two computers can transfer information bi-direction, they can both `read` and `write` data to each other. This build connectivity graphs

##### Skype

Sometimes clients are behind **NAT** which clients can send request to other servers, but server can't send request to the client, so in order for two clients to communicate, they connect to **relay** and through relay to build connection and sending data.



### 1.2 The four layers Internet model

##### Four layer hierarchy

Application -> Transport -> Network -> Link

##### Link layer

Link layer do the job to build connection to other link and routers

##### Network layer

[Data, from, to] build blocks and store data, and transfer data through link layer, through link layer, it transfers datadiagram for network to receivce and continue transfer data blocks if it is the `destination`.  

##### IP

In order for network layer to work, it must use Internet Protocol

- IP makes best-effort to deliver datagrams to the other end, but it **makes no promise**
- IP datagrams can get **lost**, can be delivered out of order and be **corrupted**, there are no guarantees

##### Transport layer

Transport layer has **TCP(Transmission control protocal)** which makes sure the data transfer in correct order and correct destination. Transport layer also has **UDP(User datagram protocal)** it can't guarantee that data is transferred properly.

##### Application layer

Application layer handles **Get** request to the server. 

##### Summary

1. Each layer only communicates with the same layer, without regarding how data get each other
2. Network layer is **Thin waist**, because it must use Internet protocal

![4layer](/Users/kuny/Downloads/notes/cuts/Network/4layer.png)



### 1.3 The IP service model

 ##### Data transfer layer

![how-data-transfer-between-layers](/Users/kuny/Downloads/notes/cuts/Network/dataTransfer.png)

##### IP service model

![ipModel](/Users/kuny/Downloads/notes/cuts/Network/ipModel.png)

Datagram: provides routing (path) to the destination

Best effort: It only drops data when necessary, e.g. the router queue is full and has to drop the next arriving data package

Unreliable: The data package may be duplicated, dropped or not send at all

Connectionless: The IP is isolate, it dones't know each other

##### Why IP service so simple

![ipSimple](/Users/kuny/Downloads/notes/cuts/Network/ipSimple.png)

Simple: If it is simple enough, it is easier to maintain, upgrade can be reliable.

End-to-end: build most features to the end, not build features on hardware to make implement correctly

Unreliable: For a real-time-chat application, we don't want data that congested or later to be retransmitted, because it may out of meaning or useless at all

Few assumptions: easy to connect through any routers and network systems 

##### IP service model



![ipServiceModel](/Users/kuny/Downloads/notes/cuts/Network/ipServiceModel.png)

It has a count, once the count == 0, stop 

Split data into two data & form two datadiagrams

Security reason when send to wrong place

New fields can be added to the headers

##### IPv4

![ipv4](/Users/kuny/Downloads/notes/cuts/Network/ipv4.png)

### 1.4 Life of a Package

##### 4 layers

**Application**: stream of data  -->**Transport**: segment of data -->**Network**: packets of data

Network layer: Responsible for delivering packets to computers, which **computerc** to deliver

Transport layer: Responsible for delivering data to applications, which **application** to deliver

##### TCP

Client -> send syn(synchronize) to Server, Server -> send syn/ack to Client, Client send ack to Server



