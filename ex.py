import numpy as np

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64)
y = np.array([[0, 1, 1, 0]], dtype=np.float64)
y_hat = np.empty((1,4), dtype=np.float64)
m = X.shape[1]


W_list = [None]
b_list = [None]
z_list = [None]
a_list = [X]

dW_list = [None]
db_list = [None]
dz_list = [None]
da_list = [None]

beta = 0.9

hided_units_nums = [2, 2, 1]
lr = 1
random_init = True

def alloc():
    for i in range(1, len(hided_units_nums)):
        if random_init:
            W_list.append(np.random.normal(0, 1, (hided_units_nums[i], hided_units_nums[i-1])))
            b_list.append(np.zeros((hided_units_nums[i], 1), np.float64))
            # b_list.append(np.random.normal(0, 0.1, (hided_units_nums[i], 1)))
        else:
            W_list.append(np.zeros((hided_units_nums[i], hided_units_nums[i-1]), dtype=np.float64))
            b_list.append(np.zeros((hided_units_nums[i], 1), np.float64))

        z_list.append(np.zeros((hided_units_nums[i], m), np.float64))
        a_list.append(np.zeros((hided_units_nums[i], m), np.float64))

        dW_list.append(np.zeros_like(W_list[i]))
        db_list.append(np.zeros_like(b_list[i]))
        dz_list.append(np.zeros_like(z_list[i]))
        da_list.append(np.zeros_like(a_list[i]))


def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


def d_sigmoid(z):
    d = sigmoid(z)*(1-sigmoid(z))
    return d


def tanh(z):
    a = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return a


def d_tanh(z):
    d = 1 - tanh(z)*tanh(z)
    return d


def forward_propagation():
    for i in range(1, len(hided_units_nums)):
        z_list[i] = np.matmul(W_list[i], a_list[i-1])+b_list[i]
        a_list[i] = sigmoid(z_list[i])
    return None


def backward_propagation():
    # dz_list[-1] = (a_list[-1] - y)
    da_list[-1] = -y/a_list[-1] + (1-y)/(1-a_list[-1])
    for i in range(len(hided_units_nums)-1, 0, -1):
        dz_list[i] = da_list[i]*d_sigmoid(z_list[i])
        dW_list[i] = beta*dW_list[i] + (1-beta) * 1/m * np.matmul(dz_list[i], a_list[i-1].transpose())
        # db_list[i] = dz_list[i]
        db_list[i] = beta*db_list[i] + (1-beta) * 1/m * np.sum(dz_list[i], axis=1, keepdims=True)
        da_list[i-1] = np.matmul(W_list[i].transpose(), dz_list[i])
    return None


def update_parameters():
    for i in range(1, len(hided_units_nums)):
        W_list[i] = W_list[i] - lr*dW_list[i]
        b_list[i] = b_list[i] - lr*db_list[i]


def print_parameters():
    for i in range(1, len(hided_units_nums)):
        print 'layer %d' %i
        print 'forward propagation'
        print W_list[i]
        print b_list[i]
        print z_list[i]
        print a_list[i]

        print 'backward propagation'
        print dW_list[i]
        print db_list[i]
        print dz_list[i]
        print da_list[i]

        print '\n'


if __name__ == '__main__':
    np.set_printoptions(8, suppress=True)

    alloc()
    for i in range(10000):
        if i % 1000 is 0:
            print 'round %d' %i
        forward_propagation()
        backward_propagation()
        update_parameters()
        # print_parameters()
        # print '\n\n\n'
    print a_list[-1]
