import random
import numpy as np
from pathlib import Path

ps = [97, 113]
ops = ['+', '-', '/']

def get_op_token(op, p):
    if op == '+':
        return p
    elif op == '-':
        return p + 1
    elif op == '/':
        return p + 2
    elif op == '=':
        return p + 3


data = []
p = ps[0]
op = ops[2]

for a in range(p):
    for b in range(p):

        if op == '+':
            c = (a + b) % p
        elif op == '-':
            c = (a - b) % p
        elif op == "/":
            if b == 0:
                continue  # skip div by zero

            inv_b = pow(b, p - 2, p)   # modular inverse of b mod p
            c = (a * inv_b) % p

        data.append((a, get_op_token(op, p), b, get_op_token('=', p), c))


# shuffle data, split into 50/40/10 train/val/test, then save

random.shuffle(data)

data_len = len(data)
data = np.array(data)

train_data = data[:int(0.5 * data_len)]
val_data = data[int(0.5 * data_len):int(0.9 * data_len)]
test_data = data[int(0.9 * data_len):]

out_dir = "/mmfs1/home/wong2/repos/493s/cse493s-spring26-hw1/arithmetic_task/data"

np.save(out_dir + "/train.npy", train_data)
np.save(out_dir + "/val.npy", val_data)
np.save(out_dir + "/test.npy", test_data)