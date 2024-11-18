import os
import sys

# add
sys.path.append("./IJCAI-AutoS")

print(sys.path)
from device_env import Evaluator, base_path
from utils.logger import info
import pickle
import torch
import warnings
import random

warnings.filterwarnings('ignore')

def convert_to_binary_list(selected_indices, list_size):
    binary_list = [0] * list_size
    for index in selected_indices:
        if 0 <= index < list_size:
            binary_list[index] = 1
    return torch.FloatTensor(binary_list)

def gen_device_selection(device_eval_):   
   
    for i in range(300):
        number_of_selections = random.randint(0, 199)
        selected_numbers = random.sample(list(range(1, 200 + 1)), number_of_selections)

        selected_numbers = convert_to_binary_list(selected_numbers, 200)
        performances = random.uniform(0, 199)
        device_eval_._store_history(selected_numbers, performances)



def process():
    device_eval = Evaluator()
    gen_device_selection(device_eval)
    file_path = f"{base_path}/history/device_env.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f'{base_path}/history/device_env.pkl', 'wb') as f: 
        pickle.dump(device_eval, f)

process()
