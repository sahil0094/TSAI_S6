# TSAI_S6

## Excel Screenshot
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/dc2909a0-b035-4d08-96e1-cf2d5e691100)

## Calculating derivatives/ Backpropagation

### Architecture
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/850a5812-3440-4221-b1b5-1c6e8429d4a0)

Here we have 1 input layer, 1 hidden layer and 1 output layer. We are provided with inital weights for each layer.<br>
In each layer starting from hidden layer, we have sigmoid function added to output for non linearity <br>
Finally we have 2 output for 2 input which are E1 & E2. We have multiplied E1 & E2 with 1/2 to make calculation <br>
easy on calculating derivatives

### Forward Propagation
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/b74cd48a-e374-416f-829e-86aad3cdd154)

We calculate output in each layer by multiplying with given weights and apply sigmoid wherever mentioned in architecture,br>
E_total is sum of E1 & E2. Here t1 & t2 are two output or true labels

### Backward Propagation

Here we have 2 layers - hidden and output layer so we will back propagate output E_total w.r.t to weights in two layer<br>
For backpropagating w.r.t last layer we will calculate derivate of E_total w.r.t w5,w6,w7,w8
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/60e967a0-227d-46bb-ae68-db016b81d7f1)

For backpropagating w.r.t hidden layer, we will calcualte derivate of E_total w.r.t w1,w2,w3,w4
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/1434c27d-a1bd-4956-a51c-bc4553873f74)

### Learning rate changes
lr=.1<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/3d96c14c-8936-4a42-b588-2e84990dc1cf)

lr=.2<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/8ac9b516-95ae-409e-bf3c-aca2ab81c11c)

lr=.5<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/c7654498-76ec-4e0e-9dab-cfcc19e71b2a)

lr=.8<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/f37c371e-6b37-4b75-8d5b-0ac4b795ef1f)

lr=1<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/7cef9423-61d9-440e-b1bd-219b210d0beb)

lr=2<br>
![image](https://github.com/sahil0094/TSAI_S6/assets/31719914/f912fcc9-54f8-4577-a71e-f0a68de9b843)

As we increase the learning rate from 0.1 to 2 , we see that loss reduces drastically and tends to 0 in less iterations
