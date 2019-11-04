
import os
import torch


def get_output_folder(parent_dir, name):
    os.makedirs(parent_dir, exist_ok=True)
    exp_id = 0
    for file_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, file_name)):
            continue
        try:
            file_id = int(file_name.split('-run')[-1])
            if file_id > exp_id:
                exp_id = file_id
        except:
            pass
    exp_id += 1

    cur_dir = os.path.join(parent_dir, name) + '-run{}'.format(exp_id)
    os.makedirs(cur_dir, exist_ok=True)
    return cur_dir


def to_tensor(array):
    tensor = torch.tensor(array).float()
    return tensor.cuda() if torch.cuda.is_available() else tensor


def to_numpy(tensor):
    return tensor.cpu().data.numpy() if torch.cuda.is_available() else tensor.data.numpy()


def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))