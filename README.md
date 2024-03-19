# FLPIE
Federated Learning Pi(e) (yummy)
> Start the Flower clients after launching the Flower server.


- For Flower server: A machine running Linux/macOS/Windows (e.g. your laptop). You can run the server on an embedded device too!
- For Flower clients (one or more): Raspberry Pi 4 (or Zero 2), or anything similar to these.

What follows is a step-by-step guide on how to setup your client/s and the server.
## Clone this example

Start with cloning this repo

```bash
git clone https://github.com/InstinctEx/FLPIE.git
```

## Setting up the server

The only requirement for the server is to have Flower installed alongside your ML framework of choice. Inside your Python environment run:

```bash
pip install flwr flwr-datasets[vision] tensorflow # to install Flower and TensorFlower
```
## Running Embedded FL with Flower

For this demo, we'll be using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a popular dataset for image classification comprised of 10 classes (e.g. car, bird, airplane) and a total of 60K `32x32` RGB images. The training set contains 50K images. The server will automatically download the dataset should it not be found in `./data`. The clients do the same. The dataset is by default split into 50 partitions (each to be assigned to a different client). This can be controlled with the `NUM_CLIENTS` global variable in the client scripts. In this example, each device will play the role of a specific user (specified via `--cid` -- we'll show this later) and therefore only do local training with that portion of the data. For CIFAR-10, clients will be training a MobileNet-v2/3 model.

You can run this example using MNIST and a smaller CNN model by passing flag `--mnist`. This is useful if you are using devices with a very limited amount of memory (e.g. RaspberryPi Zero) or if you want the training taking place on the embedded devices to be much faster (specially if these are CPU-only). The partitioning of the dataset is done in the same way.

### Start your Flower Server

On the machine of your choice, launch the server:

```bash
# Launch your server.
# Will wait for at least 2 clients to be connected, then will train for 3 FL rounds
# The command below will sample all clients connected (since sample_fraction=1.0)
# The server is dataset agnostic (use the same command for MNIST and CIFAR10)
python server.py --rounds 3 --min_num_clients 2 --sample_fraction 1.0
```

> If you are on macOS with Apple Silicon (i.e. M1, M2 chips), you might encounter a `grpcio`-related issue when launching your server. If you are in a conda environment you can solve this easily by doing: `pip uninstall grpcio` and then `conda install grpcio`.

### Start the Flower Clients

You can simulate this by manually assigning an ID to a client (`cid`) which should be an integer `[0, NUM_CLIENTS-1]`, where `NUM_CLIENTS` is the total number of partitions or clients that could participate at any point. This is defined at the top of the client files and defaults to `50`. You can change this value to make each partition larger or smaller.

```bash
# Run the default example (CIFAR-10)
python3 client.py --cid=<CLIENT_ID> --server_address=<SERVER_ADDRESS>

# Use MNIST (and a smaller model) if your devices require a more lightweight workload
python3 client.py --cid=<CLIENT_ID> --server_address=<SERVER_ADDRESS> --mnist
```

Repeat the above for as many devices as you have. Pass a different `CLIENT_ID` to each device. You can naturally run this example using different types of devices (e.g. RPi, RPi Zero, Jetson) at the same time as long as they are training the same model. If you want to start more clients than the number of embedded devices you currently have access to, you can launch clients in your laptop: simply open a new terminal and run one of the `python3 client.py ...` commands above.
