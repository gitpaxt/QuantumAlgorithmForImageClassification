import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import time
import sys
import getopt
import os
import glob

parmdict = {
  "img_size": None,
  "parity": None,
  "batch_size": None,
  "epoch": None,
  "modules": None,
  "experts": None,
  "train_samples": None,
  "test_samples": None,
  "input": None,
  "output": None,
  "user": None, 
  "seed": None,
  "ligthgpu": None,
  "adjoint": None,
  "name": None,
  "boolean_intensity": None,
  "qwtensor": None,
  "optimizer": None,
  "other_gates": None,
  "timestamp": None
}

def getParameters(argv, parmdict):
    arg_input = ""
    arg_output = ""
    arg_user = ""
    arg_img_size = ""
    arg_parity = ""
    arg_batch_size = ""
    arg_epoch = ""
    arg_modules = ""
    arg_experts = ""
    arg_train_samples = ""
    arg_test_samples = ""
    arg_seed = ""
    arg_ligthgpu = ""
    arg_adjoint = ""
    arg_name = ""
    arg_boolean_intensity = ""
    arg_qwtensor = ""
    arg_optimizer = ""
    arg_other_gates = ""
    arg_timestamp = ""

    arg_help = "{0} -i <input> -u <user> -o <output> -g <img_size> -p <parity> -b <batch_size> -e <epoch> \
-x <modules> -s <experts> -r <train_samples> -t <test_samples> -d <seed> -l \
<ligthgpu> -a <adjoint> -n <name> -m <timestamp> -q <qwtensor> -y <boolean_intensity> \
-z <optimizer> -O <other_gates>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hi:u:o:g:p:b:e:x:s:r:t:d:l:a:n:m:q:z:y:O:", ["help", "input=", \
        "user=", "output=", "img_size=", "parity=", "batch_size=", "epoch=", "modules=", "experts=", "train_samples=", \
        "test_samples=", "seed=", 'ligthgpu=', 'adjoint=', 'name=', 'timestamp=', 'qwtensor=', "boolean_intensity=",
        "test_optimizer=", "other_gates="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = arg
        elif opt in ("-u", "--user"):
            arg_user = arg
        elif opt in ("-o", "--output"):
            arg_output = arg
        elif opt in ("-g", "--img_size"):
            arg_img_size = arg
            parmdict["img_size"] = int(arg_img_size)
        elif opt in ("-p", "--parity"):
            arg_parity = arg
            parmdict["parity"] = int(arg_parity)
        elif opt in ("-b", "--batch_size"):
            arg_batch_size = arg
            parmdict["batch_size"] = int(arg_batch_size)
        elif opt in ("-e", "--epoch"):
            arg_epoch = arg
            parmdict["epoch"] = int(arg_epoch)
        elif opt in ("-x", "--modules"):
            arg_modules = arg
            parmdict["modules"] = int(arg_modules)
        elif opt in ("-s", "--experts"):
            arg_experts = arg
            parmdict["experts"] = int(arg_experts)
        elif opt in ("-r", "--train_samples"):
            arg_train_samples = arg
            parmdict["train_samples"] = int(arg_train_samples)
        elif opt in ("-t", "--test_samples"):
            arg_test_samples = arg
            parmdict["test_samples"] = int(arg_test_samples)
        elif opt in ("-d", "--seed"):
            arg_seed = arg
            parmdict["seed"] = int(arg_seed)
        elif opt in ("-l", "--ligthgpu"):
            arg_ligthgpu = arg
            if int(arg_ligthgpu) == 1:
                parmdict["ligthgpu"] = True
            else:
                parmdict["ligthgpu"] = False
        elif opt in ("-a", "--adjoint"):
            arg_adjoint = arg
            if int(arg_adjoint) == 1:
                parmdict["adjoint"] = True
            else:
                parmdict["adjoint"] = False
        elif opt in ("-n", "--name"):
            arg_name = arg
            parmdict["name"] = arg_name
        elif opt in ("-m", "--timestamp"):
            arg_timestamp = arg
            parmdict["timestamp"] = arg_timestamp
        elif opt in ("-q", "--qwtensor"):
            arg_qwtensor = arg
            parmdict["qwtensor"] = arg_qwtensor
        elif opt in ("-z", "--optimizer"):
            arg_optimizer = arg
            parmdict["optimizer"] = arg_optimizer
        elif opt in ("-O", "--other_gates"):
            arg_other_gates = int(arg)
            parmdict["other_gates"] = arg_other_gates
        elif opt in ("-y", "--boolean_intensity"):
            arg_boolean_intensity = arg
            if int(arg_boolean_intensity) == 1:
                parmdict["boolean_intensity"] = 1
            else:
                parmdict["boolean_intensity"] = 0

    print('input:', arg_input)
    print('user:', arg_user)
    print('output:', arg_output)
    print('img_size:', arg_img_size)
    print('parity:', arg_parity)
    print('batch_size:', arg_batch_size)
    print('epoch:', arg_epoch)
    print('modules:', arg_modules)
    print('experts:', arg_experts)
    print('train_samples:', arg_train_samples)
    print('test_samples:', arg_test_samples)
    print('seed:', arg_seed)
    print('ligthgpu:', arg_ligthgpu)
    print('adjoint:', arg_adjoint)
    print('name:', arg_name)
    print('timestamp:', arg_timestamp)
    print('optimizer:', arg_optimizer)
    print('other_gates:', arg_other_gates)
    print('qwtensor:', arg_qwtensor)
    print('boolean_intensity:', arg_boolean_intensity)

getParameters(sys.argv, parmdict)

OPTIMIZER = "Adam"
if parmdict["optimizer"] != None:
    if parmdict["optimizer"] == "RMSprop":
        OPTIMIZER = "RMSprop"
    elif parmdict["optimizer"] == "Adagrad":
        OPTIMIZER = "Adagrad"
    elif parmdict["optimizer"] == "Adam":
        OPTIMIZER = "Adam"
    elif parmdict["optimizer"] == "QNatGrad":
        OPTIMIZER = "QNatGrad"

print("OPTIMIZER", OPTIMIZER)

#LONG_SIM == 0 if the simulation is short.
#LONG_SIM == 1 if the simulation is long.
LONG_SIM = 1
assert(LONG_SIM == 0 or LONG_SIM == 1)

# HYBRID == 0 if we use the fully quantum model with /-rbs, with 2 blocks of repeated qubits.
HYBRID = 0

# If HYBRID == 0, I have two choices:
# RBS == 0 if we use a non-RBS gate as two qubit gate
# RBS == 1 if we use the RBS gate as two qubit gate
RBS = 0
assert(RBS == 0 or RBS == 1)

# SIMPLE_GATE == 0 if we use the two_qubit_unitary
# SIMPLE_GATE == 1 if we use the simple_two_qubit_unitary
SIMPLE_GATE = 1
assert(SIMPLE_GATE == 0 or SIMPLE_GATE == 1)

# POST_NET = 0 if the post-net is absent.
POST_NET = 0

# IMG_SIZE = 32 is the dimension of the 32x32 image obtained by padding the 28x28 image.
IMG_SIZE = 32
assert(IMG_SIZE == 32)

# PARITY = 1 if we classify parity.
# PARITY = 0 if we classify two digits.
PARITY = 1
if parmdict["parity"] != None:
    PARITY = parmdict["parity"]
assert(PARITY == 0 or PARITY == 1)

# ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY = 0 if we classify all digits.
# ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY = 1 if we classify two digits.
# ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY = 2 if we classify parity.
ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY = 1 + PARITY
assert(ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY == 1 or ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY == 2)
print("ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY = ", ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY)

# BOOLEAN_INTENSITY = 0 if we encode in the quantum state the continuous intensity of the pixels of the image.
# BOOLEAN_INTENSITY = 1 if we encode in the quantum state a boolean value (greater than average intensity) of
# the pixels of the image.
BOOLEAN_INTENSITY = 1
assert(BOOLEAN_INTENSITY == 0 or BOOLEAN_INTENSITY == 1)

# OTHER_GATES == 0 means that we use the initial version of the two gates interaction, where the interaction is strong.
# OTHER_GATES == 1 means that we use the second version of the two gates interaction, where the interaction can be
# smaller.
OTHER_GATES = 0
if parmdict["other_gates"] != None:
    if parmdict["other_gates"] == 1:
        OTHER_GATES = 1
    elif parmdict["other_gates"] == 0:
        OTHER_GATES = 0
print("OTHER_GATES", OTHER_GATES)

# LARGE_WIDTH = 0 means the deep architecture.
# LARGE_WIDTH = 1 means the shallow architecture with a large width.
LARGE_WIDTH = 1
assert(LARGE_WIDTH == 0 or LARGE_WIDTH == 1)

#LIGHTNING_GPU_QUBIT=True  lightning.gpu
#LIGHTNING_GPU_QUBIT=False lightning.qubit
LIGHTNING_GPU_QUBIT=True

# ADJOINT = True appliews. Pennylane qml.qnode diff_method is set to "adjoint". 
# ADJOINT = False appliews. Pennylane qml.qnode diff_method is set to "best".
ADJOINT=False

# Default 
NAME=False

#create log directory if not exist
path_log = "log"
if not os.path.exists(path_log): 
    os.makedirs(path_log) 

path_img = "images"
if not os.path.exists(path_img):
   os.makedirs(path_img)

path_qwt = "qwtensor"
if not os.path.exists(path_qwt):
    os.makedirs(path_qwt)

if PARITY == 0:
    FIRST_DIGIT = 0
    SECOND_DIGIT = 4

dim_image = 32

if LONG_SIM == 0:
    num_epochs = 2
    num_layer = 1
    LOSS_AVG_LENGTH = 3    #At the end, I do the average of 5 elements.
elif LONG_SIM == 1:
    num_epochs = 10
    num_layer = 3
    LOSS_AVG_LENGTH = 10    #At the end, I do the average of 5 elements.

spatial_resolution = 5
if LARGE_WIDTH == 0:
    epsilon = [6. / num_layer]*num_layer  #parameters for the nonlinear layer
elif LARGE_WIDTH == 1:
    epsilon = 0

dim_block = spatial_resolution * 2
n_blocks = 1
n_qubits = n_blocks * dim_block

q_delta = 0.5              # Initial spread of random quantum weights
n_classes = 10

if LONG_SIM == 0:
    batch_size = 2             # Number of samples for each training step
elif LONG_SIM == 1:
    batch_size = 4             # Number of samples for each training step

# Reduce the number of train images and test images
if LONG_SIM == 0:
    num_train_samples = 10 
    num_test_samples = 1
elif LONG_SIM == 1:
    num_train_samples = 300
    num_test_samples = 600

# Get the current timestamp as a string
timestamp = time.strftime("%Y%m%d%H%M%S")

PRINT_CIRCUIT = True

SEED = 5002

if parmdict["seed"] != None:
    SEED = parmdict["seed"]

if OPTIMIZER != "QNatGrad":
    torch.manual_seed(SEED)

if LARGE_WIDTH == 1:
    num_experts = 1
    num_modules = 1

num_layers_of_moe_4qubit_circuit = 3

start_time = time.time()

if parmdict["timestamp"] != None:
    timestamp = parmdict["timestamp"]


def print_config():
    l_config  =[["timestamp", True],
                ["SEED", True],
                ["LONG_SIM", True],
                ["HYBRID", True],
                ["RBS", True],
                ["SIMPLE_GATE", True],
                ["PARAMS_PER_GATE =", (LARGE_WIDTH == 0 and HYBRID == 0)],
                ["num_q_params", True],
                ["params_per_layer", (LARGE_WIDTH == 1)],
                ["num_experts", (LARGE_WIDTH == 1)],
                ["num_modules", (LARGE_WIDTH == 1)],
                ["num_train_samples", True],
                ["num_test_samples", True],
                ["POST_NET", True],
                ["ALL_DIGITS_OR_TWO_DIGITS_OR_PARITY", True],
                ["FIRST_DIGIT", (PARITY == 0)],
                ["SECOND_DIGIT", (PARITY == 0)],
                ["dim_image", True],
                ["num_epochs", True],
                ["num_layer", True],
                ["spatial_resolution", True],
                ["n_features", (HYBRID == 1)],
                ["epsilon", (LARGE_WIDTH == 0 and HYBRID == 0)],
                ["n_blocks", (LARGE_WIDTH == 0 and HYBRID == 0)],
                ["q_delta", True],
                ["n_classes", True],
                ["batch_size", True],
                ["IMG_SIZE", True],
                ["PARITY", True],
                ["LOSS_AVG_LENGTH", True],
                ["LIGHTNING_GPU_QUBIT", True],
                ["ADJOINT", True],
                ["NAME", True],
                ["BOOLEAN_INTENSITY", True],
                ["OTHER_GATES", True], # 0,1
                ["n_qubits", True]]

    for l in range(len(l_config)):
        if (l_config[l][1]):
            s="{} = {}".format(l_config[l][0], eval(l_config[l][0]))
            print(s)
            f.write(s + '\n')
    f.flush()

if parmdict["batch_size"] != None:
    batch_size = parmdict["batch_size"]

if parmdict["epoch"] != None:
    num_epochs = parmdict["epoch"]

if parmdict["experts"] != None:
    num_experts = parmdict["experts"]

if parmdict["modules"] != None:
    num_modules = parmdict["modules"]

if parmdict["train_samples"] != None:
    num_train_samples = parmdict["train_samples"]

if parmdict["test_samples"] != None:
    num_test_samples = parmdict["test_samples"]

if parmdict["ligthgpu"] != None:
    LIGHTNING_GPU_QUBIT = parmdict["ligthgpu"]

if parmdict["adjoint"] != None:
    ADJOINT = parmdict["adjoint"]

if parmdict["name"] != None:
    NAME = parmdict["name"]
else:
    NAME = f"name_{timestamp}"

if parmdict["boolean_intensity"] != None:
    BOOLEAN_INTENSITY = parmdict["boolean_intensity"]

#open output file
filename = f"log/out_{NAME}.txt"
f = open(filename, "w+")

#Load previous tensor file
check_tensor = False
if parmdict["qwtensor"] != None:
    print("qwtensor",parmdict["qwtensor"])
    if parmdict["qwtensor"] == 'last':
        list_of_files = glob.glob('qwtensor/*') # * means all if need specific format then *.csv
        qwtensor = max(list_of_files, key=os.path.getctime)
        fileqwtensorIn = f"{qwtensor}"
        print("fileqwtensorIn", fileqwtensorIn)
        check_tensor = os.path.isfile(fileqwtensorIn)

if LARGE_WIDTH == 0:
    if RBS == 1:
        PARAMS_PER_GATE = 1
    elif RBS == 0:
        if SIMPLE_GATE == 0:
            PARAMS_PER_GATE = 11
        elif SIMPLE_GATE == 1:
            PARAMS_PER_GATE = 4 
    num_q_params = num_layer*PARAMS_PER_GATE*(dim_block-1)   #The "ladder" (/) configuration of gates
elif LARGE_WIDTH == 1:
    num_layer = 2*num_experts + 1
    params_per_layer = 4*((spatial_resolution + dim_block + 1)*(spatial_resolution - 1) + (spatial_resolution + dim_block)) 
    params_per_module = params_per_layer * num_layer
    
    if OTHER_GATES == 0:
        param_per_two_qubit_gates = 4
    elif OTHER_GATES == 1:
        param_per_two_qubit_gates = 5

    num_of_block_gates = 2*(1+3*num_experts)
    if spatial_resolution == 1:
        param_per_block_gate = 1*param_per_two_qubit_gates
    elif spatial_resolution == 2:
        param_per_block_gate = 6*param_per_two_qubit_gates
    elif spatial_resolution == 5:
        param_per_block_gate = 21*param_per_two_qubit_gates
    num_of_interaction_gates = 1+2*num_experts
    param_per_interaction_gate = dim_block*param_per_two_qubit_gates

    num_q_params = num_of_block_gates*param_per_block_gate + num_of_interaction_gates*param_per_interaction_gate
    params_per_module = num_layers_of_moe_4qubit_circuit*param_per_block_gate*num_experts
    num_q_params = params_per_module*num_modules
    print("num_q_params {}".format(num_q_params))

fileqwtensorOut = f"qwtensor/qwt_{num_q_params}_{NAME}.pt"

print('---------------------')
print_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device.type)

if device.type == "cpu":
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)
    print("dev default.qubit")
else:
    if LIGHTNING_GPU_QUBIT:
        dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None, torch_device='cuda')
        print("dev default.qubit.torch")
    else:
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
        print("dev lightning.qubit")

torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=180, sci_mode=False)

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

if PARITY == 0:
    def filter_digits(dataset):
        filtered_data = []
        for data, target in dataset:
          if target == FIRST_DIGIT or target == SECOND_DIGIT:
                filtered_data.append((data, target))
        return filtered_data

    filtered_trainset = filter_digits(mnist_trainset)
    filtered_testset = filter_digits(mnist_testset)

if PARITY == 0:
    len_train = len(filtered_trainset)
    len_test = len(filtered_testset)
elif PARITY == 1:
    len_train = 60000
    len_test = 10000
    filtered_trainset = mnist_trainset
    filtered_testset = mnist_testset

mnist_trainset, _ = random_split(filtered_trainset,[num_train_samples, len_train - num_train_samples])
mnist_testset, _ = random_split(filtered_testset,[num_test_samples, len_test - num_test_samples])

#resize images
def resize_images(mnist_trainset, mnist_testset):
    class CustomDataset(Dataset):
        def __init__(self, t1):
            self.data = t1
    
        def __len__(self):
            # Return the total number of samples in your dataset
            return len(self.data)
    
        def __getitem__(self, indexm):
            # Return a specific sample (tuple of tensor and integer) based on the index
            return self.data[indexm]

    torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=180, sci_mode=False)

    img_sets = [mnist_trainset, mnist_testset]
    for img_set in img_sets:
        if img_set == mnist_trainset:
            mnist_trainset_list = []
            img_list = mnist_trainset_list
        elif img_set == mnist_testset:
            mnist_testset_list = []
            img_list = mnist_testset_list
        else:
            print("ERROR")
            exit()
        for i in range(len(img_set)):
            index_to_replace = 1
            x = img_set[i][0]
            
            # Shape of x: (2, 1, 28, 28). This and the following comments are for the case of a 28x28 image, with batch size = 2.
            x = torch.squeeze(x, 1)
            # Shape of x: (2, 28, 28)
            desired_shape = (batch_size, 32, 32)
    
            # Calculate the amount of padding needed along each dimension
            pad_height = desired_shape[1] - x.shape[1]
            pad_width = desired_shape[2] - x.shape[2]
            half_pad_height = int(pad_height/2)
            half_pad_width = int(pad_width/2)

            # Pad the input tensor to achieve the desired shape
            x = torch.nn.functional.pad(x, (half_pad_width, half_pad_width, half_pad_height, half_pad_height))

            # Shape of x: (2, 32, 32)
            if BOOLEAN_INTENSITY == 1:
                avg = torch.mean(x)
                x = torch.where(x > avg, torch.tensor(1.0), torch.tensor(-1.0))

            # Shape of x: (2, 8, 8)
            nt = (x, img_set[i][1])
            q_input = x
        
            my_tuple = (q_input, img_set[i][1])
            img_list.append(my_tuple)

    mnist_trainset_data = CustomDataset(mnist_trainset_list)
    mnist_testset_data = CustomDataset(mnist_testset_list)

    ind_1 = [i for i in range(0, len(mnist_trainset_data))]
    mnist_trainset_l = Subset(mnist_trainset_data, indices=ind_1)
    ind_1 = [i for i in range(0, len(mnist_testset_data))]
    mnist_testset_l = Subset(mnist_testset_data, indices=ind_1)

    return (mnist_trainset_l, mnist_testset_l)

mnist_trainset, mnist_testset = resize_images(mnist_trainset, mnist_testset)

dataloaders = {
    "train": DataLoader(mnist_trainset, batch_size=batch_size, shuffle=False),
    "validation": DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
}

dataset_sizes = {
        "train": num_train_samples,
        "validation": num_test_samples
}

if (ADJOINT):
    diff_method='adjoint'
else:
    diff_method='best'
if OPTIMIZER == "QNatGrad":
    print("interface='autograd'")
    interface='autograd'
else:
    interface='torch'
    print("interface='torch'")

@qml.qnode(dev, interface=interface, diff_method=diff_method)
@qml.transforms.merge_amplitude_embedding
def circuit(q_weights, q_input, epsilon, layer_index):
    #rotation beamsplitter
    def rbs(theta,wires):
        qml.CNOT([wires[0],wires[1]])
        qml.CRY(theta*2,[wires[0],wires[1]])
        qml.CNOT([wires[0],wires[1]])
    
    def two_qubit_unitary(theta, k, wires):
        qml.RZ(theta[k+0],wires[0])
        qml.RY(theta[k+1],wires[0])
        qml.RZ(theta[k+2],wires[1])
        qml.RY(theta[k+3],wires[1])
        qml.CNOT([wires[1],wires[0]])
        qml.RZ(theta[k+4],wires[0])
        qml.RY(theta[k+5],wires[1])
        qml.CNOT([wires[0],wires[1]])
        qml.RY(theta[k+6],wires[1])
        qml.CNOT([wires[1],wires[0]])
        qml.RZ(theta[k+7],wires[0])
        qml.RY(theta[k+8],wires[0])
        qml.RZ(theta[k+9],wires[1])
        qml.RY(theta[k+10],wires[1])

    def simple_two_qubit_unitary(theta, k, wires):
        if OTHER_GATES == 0:
            # The elementary 2-qubit gate used.
            qml.RX(theta[k+0],wires[0])
            qml.RX(theta[k+1],wires[1])
            qml.CNOT([wires[1],wires[0]])
            qml.RY(theta[k+2],wires[0])
            qml.RY(theta[k+3],wires[1])
        elif OTHER_GATES == 1:
            # Modified version.
            qml.RZ(theta[k+0],wires[0])
            qml.RZ(theta[k+1],wires[1])
            qml.ctrl(qml.RZ, (wires[0]), control_values=(1))(theta[k+2], wires=wires[1])
            qml.RY(theta[k+3],wires[0])
            qml.RY(theta[k+4],wires[1])

    def block_gate(q_weights, num_experts, spatial_resolution, index_of_block_gate):
        # the index of the block gate starts from 0.
        # Now we determine the first_qubit of the block gate (i.e. the block of gates)
        index_of_start_second_series = 2*(1+num_experts)
        index_of_start_third_series = 2*(1+2*num_experts)
        
        first_qubit = 0

        # Now we determine the first parameter k to be used
        param_per_two_qubit_gate = 4
        if spatial_resolution == 1:
            param_per_block_gate = param_per_two_qubit_gate
        elif spatial_resolution == 2:
            param_per_block_gate = 6*param_per_two_qubit_gate
        elif spatial_resolution == 5:
            param_per_block_gate = 20*param_per_two_qubit_gate

        k = index_of_block_gate*param_per_block_gate

        # Now we apply the block gate
        if spatial_resolution == 1:
            simple_two_qubit_unitary(q_weights, k, [first_qubit, first_qubit+1])
        elif spatial_resolution == 2:
            simple_two_qubit_unitary(q_weights, k+0*param_per_two_qubit_gate, [first_qubit+0, first_qubit+1])
            simple_two_qubit_unitary(q_weights, k+1*param_per_two_qubit_gate, [first_qubit+2, first_qubit+3])
            simple_two_qubit_unitary(q_weights, k+2*param_per_two_qubit_gate, [first_qubit+1, first_qubit+3])
            simple_two_qubit_unitary(q_weights, k+3*param_per_two_qubit_gate, [first_qubit+0, first_qubit+2])
            simple_two_qubit_unitary(q_weights, k+4*param_per_two_qubit_gate, [first_qubit+0, first_qubit+1])
            simple_two_qubit_unitary(q_weights, k+5*param_per_two_qubit_gate, [first_qubit+2, first_qubit+3])
        elif spatial_resolution == 5:
            simple_two_qubit_unitary(q_weights, k+0*param_per_two_qubit_gate, [first_qubit+6, first_qubit+7])
            simple_two_qubit_unitary(q_weights, k+1*param_per_two_qubit_gate, [first_qubit+8, first_qubit+9])
            simple_two_qubit_unitary(q_weights, k+2*param_per_two_qubit_gate, [first_qubit+7, first_qubit+9])
            simple_two_qubit_unitary(q_weights, k+3*param_per_two_qubit_gate, [first_qubit+6, first_qubit+8])

            simple_two_qubit_unitary(q_weights, k+4*param_per_two_qubit_gate, [first_qubit+4, first_qubit+5])
            simple_two_qubit_unitary(q_weights, k+5*param_per_two_qubit_gate, [first_qubit+6, first_qubit+7])
            simple_two_qubit_unitary(q_weights, k+6*param_per_two_qubit_gate, [first_qubit+8, first_qubit+9])
            simple_two_qubit_unitary(q_weights, k+7*param_per_two_qubit_gate, [first_qubit+5, first_qubit+7])
            simple_two_qubit_unitary(q_weights, k+8*param_per_two_qubit_gate, [first_qubit+4, first_qubit+6])

            simple_two_qubit_unitary(q_weights, k+9*param_per_two_qubit_gate, [first_qubit+2, first_qubit+3])
            simple_two_qubit_unitary(q_weights, k+10*param_per_two_qubit_gate, [first_qubit+4, first_qubit+5])
            simple_two_qubit_unitary(q_weights, k+11*param_per_two_qubit_gate, [first_qubit+6, first_qubit+7])
            simple_two_qubit_unitary(q_weights, k+12*param_per_two_qubit_gate, [first_qubit+3, first_qubit+5])
            simple_two_qubit_unitary(q_weights, k+13*param_per_two_qubit_gate, [first_qubit+2, first_qubit+4])

            simple_two_qubit_unitary(q_weights, k+14*param_per_two_qubit_gate, [first_qubit+0, first_qubit+1])
            simple_two_qubit_unitary(q_weights, k+15*param_per_two_qubit_gate, [first_qubit+2, first_qubit+3])
            simple_two_qubit_unitary(q_weights, k+16*param_per_two_qubit_gate, [first_qubit+4, first_qubit+5])
            simple_two_qubit_unitary(q_weights, k+17*param_per_two_qubit_gate, [first_qubit+1, first_qubit+3])
            simple_two_qubit_unitary(q_weights, k+18*param_per_two_qubit_gate, [first_qubit+0, first_qubit+2])

            simple_two_qubit_unitary(q_weights, k+19*param_per_two_qubit_gate, [first_qubit+0, first_qubit+1])
            simple_two_qubit_unitary(q_weights, k+20*param_per_two_qubit_gate, [first_qubit+2, first_qubit+3])
        return

    def interaction_gate(q_weights, num_experts, spatial_resolution, index_of_interaction_gate):
        # The index of the interaction gate starts from 0.
        # Now we determine the first_qubit of the interaction gate
        index_of_start_second_series = 1+num_experts
        
        if index_of_interaction_gate < index_of_start_second_series:
            first_qubit = index_of_interaction_gate*2*dim_block
        else:
            first_qubit = (index_of_interaction_gate-index_of_start_second_series)*2*dim_block+dim_block
        
        # Now we determine the first parameter k to be used
        param_per_two_qubit_gate = 4
        param_per_interaction_gate = dim_block*param_per_two_qubit_gate

        num_of_block_gates = 2*(1+3*num_experts)
        
        k = num_of_block_gates + index_of_interaction_gate*param_per_interaction_gate
        
        # Now we apply the interaction gate
        if spatial_resolution == 1:
            for i in range(dim_block):
                simple_two_qubit_unitary(q_weights, k+i, [first_qubit+i, first_qubit+i+dim_block])
        return

    def nonlinear_layer(epsilon_element, wires):
        qml.IsingXX(epsilon_element, wires)

    def slash_rbs(thetas, epsilon, layer, wires):
        #The number of parameters is lw-1, where lw is the number of wires.
        k = (layer-1)*PARAMS_PER_GATE*(dim_block-1)
        lw = len(wires)
        # blw is the block length wires, i.e. the length of the wires of a single block.
        if n_blocks == 2:
            blw = int(lw / 2)
        elif n_blocks == 1:
            blw = lw
        if RBS == 1:
            for i in range(blw-1):
                rbs(thetas[k], [wires[blw-2-i],wires[blw-1-i]])
                rbs(thetas[k], [wires[blw-2-i+dim_block],wires[blw-1-i+dim_block]])
                nonlinear_layer(epsilon[layer-1], [wires[blw-1-i], wires[blw-1-i+dim_block]])
                k += 1
            assert(k == layer*(dim_block-1))
        if RBS == 0:
            for i in range(blw-1):
                if SIMPLE_GATE == 0:
                    two_qubit_unitary(thetas, k, [wires[blw-2-i],wires[blw-1-i]])
                    if n_blocks == 2:
                        two_qubit_unitary(thetas, k, [wires[blw-2-i+dim_block],wires[blw-1-i+dim_block]])
                elif SIMPLE_GATE == 1:
                    simple_two_qubit_unitary(thetas, k, [wires[blw-2-i],wires[blw-1-i]])
                    if n_blocks == 2:
                        simple_two_qubit_unitary(thetas, k, [wires[blw-2-i+dim_block],wires[blw-1-i+dim_block]])
                if n_blocks == 2:
                    nonlinear_layer(epsilon[layer-1], [wires[blw-1-i], wires[blw-1-i+dim_block]])
                k += PARAMS_PER_GATE
            if n_blocks == 2:
                nonlinear_layer(epsilon[layer-1], [wires[0], wires[dim_block]])
            assert(k == PARAMS_PER_GATE*layer*(dim_block-1))

    # Encoding
    wires1 = []
    for i in range(spatial_resolution):
        wires1.append(2*i)
    for i in range(spatial_resolution):
        wires1.append(2*i+1)

    if OPTIMIZER == "QNatGrad":
        state1 = pnp.zeros([2**(n_qubits//n_blocks)], dtype=pnp.cfloat, device=device)
    else:
        state1 = torch.zeros([2**(n_qubits//n_blocks)], dtype=torch.cfloat, device=device)
    
    for i_x in range(2**spatial_resolution):
        for i_y in range(2**spatial_resolution):
            state1[i_x*2**spatial_resolution+i_y]=q_input[i_x,i_y]+0.0001
    if OPTIMIZER == "QNatGrad":
        state1 = pnp.array(state1)
        state1 = state1/pnp.linalg.norm(state1)
    else:
        state1 = state1/torch.linalg.norm(state1)

    qml.AmplitudeEmbedding(features=state1, wires=wires1, normalize=True)

    wires = list(range(n_qubits))
    
    if LARGE_WIDTH == 0:
        for layer in range(1,num_layer+1):
            slash_rbs(q_weights, epsilon, layer, wires)
            qml.Barrier(wires=wires, only_visual=True)
    elif LARGE_WIDTH == 1:
        # The formula is:
        # params_per_layer = 4*((spatial_resolution + len(wires1) + 1)*(spatial_resolution - 1) + (spatial_resolution + len(wires1))) 
        
        # layer_index is the index of the expert in which we are. It starts from 0.
        for index in range(num_layers_of_moe_4qubit_circuit):
            block_gate(q_weights, num_experts, spatial_resolution, 2*layer_index+index)

    if LARGE_WIDTH == 0:
        result = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    elif LARGE_WIDTH == 1:
        # In the expert, I take only the n_qubits/2 in the center. Or all n_qubits in the MoE 4qubit circuit case.
        result = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if OPTIMIZER == "QNatGrad":
       return result
    else:
       return result

def cost(q_weights, inputs, epsilon, weight, bias, epoch, right_predictions_counter, label_counter, confusion_matrix, layer_index):
    if OPTIMIZER == "QNatGrad":
        q_output = pnp.zeros((min(batch_size,len(labels)), int(n_qubits/2)))
    else:
        q_output = torch.zeros(min(batch_size, len(labels)), n_qubits, device=device)

    print("q_weights:", q_weights)
    print("type(q_weights):", type(q_weights))
    
    q_input = inputs

    if LARGE_WIDTH == 1:
        if OPTIMIZER == "QNatGrad":
            final_output = pnp.zeros(len(labels))
        else:
            final_output = torch.zeros(len(labels), requires_grad=True, device=device)

        for j in range(num_experts):
            c = []
            # The min below is to treat the case in which the number of images is not divisible by the batch size.
            for image in range(len(labels)):
                layer_index = j
                c.append(circuit(q_weights, q_input[image][0], epsilon, layer_index))
            for image in range(len(labels)):
                if OPTIMIZER == "QNatGrad":
                    new_c = qml.math.toarray(c)
                    print("len(c) = ", len(c))
                qubits_seen = n_qubits
                for i in range(qubits_seen):
                    if OPTIMIZER == "QNatGrad":
                        q_output[image][i] = qml.math.toarray(new_c[image][i])
                    else:
                        q_output[image][i] = c[image][i]

            if OPTIMIZER == "QNatGrad":
                final_output_temp = pnp.sum(q_output, 1)*8.0/(pnp.sqrt(num_experts))
            else:
                q_output_not_normalized = torch.sum(q_output, 1)
                factor_batchnorm = torch.sum(torch.square(q_output_not_normalized), 0)
                print("factor_batchnorm = ", torch.sqrt(factor_batchnorm))
                
                final_output_temp = q_output_not_normalized / 2.
            final_output = final_output + final_output_temp
        
        print("FINAL OUTPUT = ", final_output)

        if OPTIMIZER == "QNatGrad":
            loss = (pnp.square(final_output - labels.detach().numpy())).mean()
            preds = pnp.sign(final_output)*1.
            count_equal_elements = pnp.sum(preds == labels).item()
        else:
            loss = criterion(final_output, labels)
            preds = torch.sign(final_output)*1.
            count_equal_elements = torch.sum(preds == labels).item()
        right_predictions_counter += count_equal_elements
        label_counter += len(labels)
        cl_output = 0
        confusion_matrix = 0
        if OPTIMIZER == "QNatGrad":
            return loss
        else:
            return loss, right_predictions_counter, preds, label_counter, confusion_matrix, cl_output
    
    if LARGE_WIDTH == 0:
        c = []
        for image in range(batch_size):
            c.append(circuit(q_weights, q_input[image][0], epsilon, layer_index))
            for i in range(n_qubits):
                q_output[image][i] = c[image][i]
        cl_output = q_output.detach() + 0.0
        cl_output.requires_grad_(True)
        final_output = torch.sum(q_output, 1)/6.
        preds = torch.sign(final_output) * 1.
    
        loss = criterion(final_output, labels)
        
        count_equal_elements = torch.sum(preds == labels).item()
        print("At epoch " + "{:3}".format(epoch) + " the prediction is " + str(preds) + " while the true label is " + str(labels) + " . Success ratio: " + "{:4}".format(count_equal_elements) + "/" + "{:4}".format(batch_size) + " [" + "{:3.2f}".format(100. * count_equal_elements/batch_size) + "]" + "\n")
        f.write("At epoch " + "{:3}".format(epoch) + " the prediction is " + str(preds) + " while the true label is " + str(labels) + " . Success ratio: " + "{:4}".format(count_equal_elements) + "/" + "{:4}".format(batch_size) + " [" + "{:3.2f}".format(100. * count_equal_elements/batch_size) + "]" + "\n")
        f.flush()
        right_predictions_counter += count_equal_elements
    
        cl_output = 0
        
        if OPTIMIZER == "QNatGrad":
            return loss
        else:
            return loss, right_predictions_counter, confusion_matrix, cl_output

if check_tensor:
    qwt = torch.load(fileqwtensorIn)
    print("load q_weights.size()  = ", qwt.size())
    print("load q_weights  = ", qwt)
    qwt = qwt.to(device)
    print("qwt\n", qwt)

    q_weights = qwt
else:
    q_weights = Variable(q_delta * torch.randn(num_q_params, device=device), requires_grad=True)

q_input = Variable(torch.randn(dim_image, dim_image, device=device), requires_grad=False)

if OPTIMIZER == "RMSprop":
    print("opt RMSprop")
    opt = torch.optim.RMSprop([q_weights], lr = 0.003, alpha = 0.9)
elif  OPTIMIZER == "Adagrad":
    print("opt Adagrad")
    opt = torch.optim.Adagrad([q_weights], lr = 0.003)
else:
    print("opt Adam")
    opt = torch.optim.Adam([q_weights], lr = 0.003)

criterion = nn.MSELoss()

if PRINT_CIRCUIT == True:
    print("q_input = ", q_input)
    drawer = qml.draw(circuit)
    print(drawer(q_weights, q_input, epsilon, 0))

if OPTIMIZER == "QNatGrad":
    #init_params =          pnp.array([0.432, -0.123, 0.543, 0.233], requires_grad=True)

    # Approximation in which only diagonal elements are computed.
    # Alternatives are "None" and "block-diag".
    approx='"diag"'

    rng = pnp.random.default_rng(seed=42)  # make the results reproducable
    q_weights = q_delta * rng.random([num_q_params], requires_grad=True)
    opt = qml.QNGOptimizer(0.1, approx=approx)
    print("OPTIMIZER", OPTIMIZER)
    phase = 'train'
    print("q_weights: out\n", q_weights)

    for epoch in range(num_epochs + 1):
        image_in_epoch_index = 0
        for inputs, labels in dataloaders[phase]:
            q_input = inputs
            
            numpy_q_input =  q_input.detach().numpy()
            q_input = pnp.array(numpy_q_input, requires_grad=False)
            print("q_input.requires_grad", q_input.requires_grad)

            for module in range(num_modules):
                print("epoch {} labels {} module {} \n---\ninputs:\n{}\n---".format(epoch,labels,module,inputs))
                right_predictions_counter = 0
                label_counter = 0
                confusion_matrix = 0
                
                cost_fn = lambda p: cost(p, q_input, epsilon, 0, 0, epoch, right_predictions_counter, label_counter, confusion_matrix, layer_index)
                
                print("num_experts",num_experts)
                print("min(batch_size,len(labels))", min(batch_size,len(labels)))

                str_cmd='lambda p: '
                str_par=""
                print("str_cmd", str_cmd)
                for j in range(num_experts):
                    for image in range(min(batch_size,len(labels))):
                        if ((j == num_experts - 1) and (image == min(batch_size,len(labels)) - 1)):
                            str_cmd += 'qml.metric_tensor(circuit, approx=' + approx + ')(p, q_input[' + str(image) + '][0], epsilon, ' + str(j) + ')'
                            str_cmd += str_par
                        else:
                            str_cmd += 'sum(qml.metric_tensor(circuit, approx=' + approx + ')(p, q_input[' + str(image) + '][0], epsilon, ' + str(j) + '),'
                            str_par += ')'

                epsilon = 0
                layer_index = 0
                x1=circuit(q_weights, q_input[0][0], epsilon, layer_index)
                print("x1",x1)
                epsilon = 0
                layer_index = 0
                x2=circuit(q_weights, q_input[1][0], epsilon, layer_index)
                print("x2",x2)

                str_cmd='lambda p: sum(qml.metric_tensor(circuit, approx=' + approx + ')(p, (q_input[1][0] + q_input[0][0]), epsilon, 0),' + \
                                      'qml.metric_tensor(circuit, approx=' + approx + ')(p, (q_input[0][0] + q_input[1][0]), epsilon, 0))' + \
                                      '/(num_experts * min(batch_size,len(labels)))'

                str_cmd='lambda p: sum(qml.metric_tensor(circuit, approx=' + approx + ')(p, q_input[1][0], epsilon, 0),' + \
                                      'qml.metric_tensor(circuit, approx=' + approx + ')(p, q_input[0][0], epsilon, 0))' + \
                                      '/(num_experts * min(batch_size,len(labels)))'

                print ("str_cmd >>>{}<<<".format(str_cmd))

                metric_fn_test = eval(str_cmd)
                print("metric_fn_test = ", metric_fn_test)
                

                print("q_weights.shape", q_weights.shape)
                print("q_weights", q_weights)
                print("q_weights.shape: ", q_weights.shape)
                a1 = metric_fn_test(q_weights)
                print("type(a1)", type(a1))
                print("a1", a1)
                print("a1.shape", a1.shape)
                for i_1 in range(a1.shape[0]):
                   for j_1 in range(a1.shape[1]):
                       print(" {:3.2f}".format(a1[i_1,j_1]),  end="")
                   print("")

                print("q_weights:\n", q_weights)
                my_temps = opt.step_and_cost(cost_fn, q_weights, metric_tensor_fn=metric_fn_test)
                print("labels = ", labels)
                other_index = 0
                
                for index in range(len(my_temps)):
                    if index == 0:
                        print("my_temps[", index, "].shape = ", my_temps[index].shape)
                        print("my_temps[", index, "] = ", my_temps[index])
                        q_weights = my_temps[index]
                print("ind1 ind2", module*params_per_module, (module+1)*params_per_module)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
    f.close()
    exit() 

print("OPTIMIZER", OPTIMIZER)
print("q_weights:", q_weights)

loss_history = []
for epoch in range(num_epochs + 1):
    # Iterate over data.
    phases = ['train', 'validation']

    for phase in phases:
        right_predictions_counter = 0
        preds = []
        label_counter = 0
        confusion_matrix  = [[0] * 10 for _ in range(10)]   #the i-th digits is taken as the j-th digit.
        n_batches = dataset_sizes[phase] // batch_size
        # For federal predictions I mean the predictions in which each experts has one vote given by its internal majority.
        # It was not used for train or test. It was defined to look for eventual anomalies.
        right_federal_predictions_counter = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            count_federal_equal_elements = 0
            federal_preds = torch.zeros(len(labels), device=device)
            
            if PARITY == 0:
                labels = -1 + ((labels - FIRST_DIGIT) * 2./(SECOND_DIGIT-FIRST_DIGIT))
            elif PARITY == 1:
                labels = ((labels%2)*2)-1.
            
            for module in range(num_modules):
                if OPTIMIZER != "QNatGrad":
                   opt.zero_grad()
                if LARGE_WIDTH == 0:
                    loss, right_predictions_counter, confusion_matrix, _ = cost(q_weights[module*params_per_module:(module+1)*params_per_module], inputs, epsilon, 0, 0, epoch, right_predictions_counter, confusion_matrix, 0)
                elif LARGE_WIDTH == 1:
                    print("q_weights:", q_weights)
                    print("q_weights[module*params_per_module:(module+1)*params_per_module]", q_weights[module*params_per_module:(module+1)*params_per_module])
                    print("module*params_per_module {} (module+1)*params_per_module {}", module*params_per_module, (module+1)*params_per_module)
                    
                    # The other line is for the QNatGrad case
                    loss, right_predictions_counter, preds, label_counter, confusion_matrix, _ = cost(q_weights[module*params_per_module:(module+1)*params_per_module], inputs, epsilon, 0, 0, epoch, right_predictions_counter, label_counter, confusion_matrix, 0)
                    federal_preds += preds
                loss_history.append(loss.item())
                
                if phase == 'train':
                    print("The train loss is equal to: ", loss)
                    print("q_weights         = ", q_weights)
                    if (epoch >= 0):
                        loss.backward()
                    print("q_weights.grad    = ", q_weights.grad)
                    if (q_weights.grad != None):
                        print("q_weights.grad.shape    = ", q_weights.grad.shape)
                        print("norm of q_weights.grad    = ", torch.norm(q_weights.grad))
                        opt.step()
                print("q_weights\n[{}]".format(q_weights))
            print("federal_preds {}",federal_preds) 
            print("labels        {}",labels) 
            federal_preds = torch.sign(federal_preds)*1.
            print("federal_preds {}",federal_preds) 
            print("labels        {}",labels) 
            print("count_federal_equal_elements {}".format(count_federal_equal_elements)) 
            print("right_predictions_counter {}".format(right_predictions_counter)) 
            count_federal_equal_elements = torch.sum(federal_preds == labels).item()
            right_federal_predictions_counter += count_federal_equal_elements
            print("count_federal_equal_elements {}".format(count_federal_equal_elements)) 
            print("right_predictions_counter {}".format(right_predictions_counter)) 
        
        if phase == 'train':
            print("The total number of right predictions with train_loss at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_predictions_counter) + "/" + "{:5}".format(num_modules*num_train_samples) + " [" + "{:3.2f}".format(100. * right_predictions_counter/(num_modules*num_train_samples)) + "%] with label_counter = " + "{}".format(label_counter))
            f.write("The total number of right predictions with train_loss at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_predictions_counter) + "/" + "{:5}".format(num_modules*num_train_samples) + " [" + "{:3.2f}".format(100. * right_predictions_counter/(num_modules*num_train_samples)) + "%]" + "\n")
            f.flush()
            print("The total number of right federal predictions with train loss at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_federal_predictions_counter) + "/" + "{:5}".format(num_train_samples) + " [" + "{:3.2f}".format(100. * right_federal_predictions_counter/(num_train_samples)) + "%] with label_counter = ")
            f.write("The total number of right federal predictions with train loss at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_federal_predictions_counter) + "/" + "{:5}".format(num_train_samples) + " [" + "{:3.2f}".format(100. * right_federal_predictions_counter/(num_train_samples)) + "%]" + "\n")
            f.flush()
        if phase == 'validation':
            print("The total number of right predictions with TEST_LOSS at epoch  " + "{:5}".format(epoch) + " is " + "{:5}".format(right_predictions_counter) + "/" + "{:5}".format(num_modules*num_test_samples) + " [" + "{:3.2f}".format(100. * right_predictions_counter/(num_modules*num_test_samples)) + "%] with label_counter = " + "{}".format(label_counter))
            f.write("The total number of right predictions with TEST_LOSS at epoch  " + "{:5}".format(epoch) + " is " + "{:5}".format(right_predictions_counter) + "/" + "{:5}".format(num_modules*num_test_samples) + " [" + "{:3.2f}".format(100. * right_predictions_counter/(num_modules*num_test_samples)) + "%]" + "\n")
            f.flush()
            print("The total number of right federal predictions with TEST  LOSS at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_federal_predictions_counter) + "/" + "{:5}".format(num_test_samples) + " [" + "{:3.2f}".format(100. * right_federal_predictions_counter/(num_test_samples)) + "%] with label_counter = ")
            f.write("The total number of right federal predictions with TEST  LOSS at epoch " + "{:5}".format(epoch) + " is " + "{:5}".format(right_federal_predictions_counter) + "/" + "{:5}".format(num_test_samples) + " [" + "{:3.2f}".format(100. * right_federal_predictions_counter/(num_test_samples)) + "%]" + "\n")
            f.flush()
       
loss_averaged_history = []
for i in range(max(len(loss_history) - LOSS_AVG_LENGTH + 1, 0)):
    val = 0
    for j in range(LOSS_AVG_LENGTH):
        val += loss_history[i + j]
    loss_averaged_history.append(val/LOSS_AVG_LENGTH)

image_indices = range(len(loss_history))
image_indices_averaged = range(LOSS_AVG_LENGTH // 2, LOSS_AVG_LENGTH //2 + len(loss_averaged_history))
assert(len(image_indices_averaged) == len(loss_averaged_history))

# Plot the list of loss values
plt.figure(figsize=(8, 6))
plt.plot(image_indices, loss_history, marker='o', linestyle='-', color='b')
plt.plot(image_indices_averaged, loss_averaged_history, marker='o', linestyle='-', color='r')
plt.title('Loss per Image')
plt.xlabel('Image Index')
plt.ylabel('Loss')
plt.grid(True)
#plt.show()

torch.save(q_weights, fileqwtensorOut)
print("save q_weights.size() = ", q_weights.size())
print("save q_weights = ", q_weights)

# Define the filename with the timestamp
filename = f"loss_graph_24_par_1_long_sim_1_time_{timestamp}"

plt.savefig('./' + path_img + '/' + filename + '.png', format='png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.4f} seconds")
f.close()
