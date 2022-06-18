import copy
import torch
import numpy as np
import time
from matplotlib import pyplot as plt
from Plot import PlotClass


dtype = torch.float
device = torch.device("cuda:0")
# device = torch.device("cpu")
trainset_percentage = 80
nepoch = 500
nnode = 32
nlayer = 5
nbatch = 100
input = np.load('./input.npy')
# input = np.delete(input, (0,1,2,3), axis=1)
# input = np.delete(input, (7,8), axis=1)
input = np.delete(input, (0,1,2,3,7,8), axis=1)
lable = np.load('./lable.npy').astype(int)

(datanumber, inputsize) = input.shape
(datanumber, lablesize) = lable.shape
trainnumber = int(round(datanumber * trainset_percentage / 100))
random_index = np.load('idx.npy')
# random_index = np.random.permutation(np.arange(datanumber))
# np.save('idx.npy',random_index)
train_input = input[random_index[:trainnumber + 1], :]
train_lable = lable[random_index[:trainnumber + 1], :]
test_input = input[random_index[trainnumber + 1:], :]
test_lable = lable[random_index[trainnumber + 1:], :]

# det = True
# while det:
#     random_index = np.random.permutation(np.arange(datanumber))
#     test_lable = lable[random_index[trainnumber + 1:], :]
#     tt = np.sum(test_lable, axis=0)
#     if np.prod(tt, dtype=float) > 0.001:
#         det = False
#     else:
#         pass

train_input = torch.tensor(train_input, dtype=dtype, device=device)
train_lable = torch.tensor(train_lable, dtype=dtype, device=device)
test_input = torch.tensor(test_input, dtype=dtype, device=device)
test_lable = torch.tensor(test_lable, dtype=dtype, device=device)

print('n_data = {0:d}, n_train = {1:d}, n_test = {2:d}, n_input = {3:d}, n_output = {4:d}, batch = {5:.2f} * {6:d} batches'
      .format(datanumber, trainnumber, datanumber - trainnumber, inputsize, lablesize, trainnumber / nbatch, nbatch))

traindataset = torch.utils.data.TensorDataset(train_input, train_lable)
testdataset = torch.utils.data.TensorDataset(test_input, test_lable)
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=int(trainnumber / nbatch), shuffle=True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=len(test_input))

network_queue = []
for i in range(nlayer):
    if i == 0:
        network_queue.append(torch.nn.Linear(inputsize, nnode))
        # network_queue.append(torch.nn.BatchNorm1d(nnode))
        network_queue.append(torch.nn.ReLU())
    else:
        network_queue.append(torch.nn.Linear(nnode, nnode))
        # network_queue.append(torch.nn.BatchNorm1d(nnode))
        network_queue.append(torch.nn.ReLU())
network_queue.append(torch.nn.Linear(nnode, lablesize))

net = torch.nn.Sequential(*network_queue).to(device)

def weight_init_kaiming_uniform(submodule):
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(submodule.weight)
        submodule.bias.data.fill_(0.00)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

net.apply(weight_init_kaiming_uniform)

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)


time_start = time.time()
train_losses = []
test_losses = []
correct_hist = []
best_test_loss = 1000
best_correct = 0
for epoch in range(nepoch):
    running_loss = 0.0
    net.train()
    for xx, yy in traindataloader:
        y_pred = net(xx)
        train_loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
        optimizer.step()
        running_loss += train_loss.item() * len(xx)
    train_losses.append(running_loss / len(train_input))
    net.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xx, yy in testdataloader:
            y_pred = net(xx)
            test_loss = loss_fn(y_pred, yy)
            running_loss += test_loss.item() * len(xx)
            correct += (y_pred.argmax(1) == yy.argmax(1)).type(torch.float).sum().item()
        test_losses.append(running_loss / len(test_input))
        correct_hist.append(correct / len(test_input) * 100)
    # if test_losses[-1] < best_test_loss:
    #     best_test_loss = test_losses[-1]
    #     best_model = copy.deepcopy(net)
    if correct_hist[-1] > best_correct:
        best_correct = correct_hist[-1]
        best_model = copy.deepcopy(net)
    if (epoch + 1) % 1 == 0:
        print('iter = {0:g}, time = {1:3.2f}s/1iter, test_loss = {2:3.2e}, train_loss = {3:3.2e}, test_acc = {4:0.1f}%'
              .format(epoch + 1, time.time() - time_start, test_losses[-1], train_losses[-1], correct_hist[-1]))
        time_start = time.time()
for xx, yy in testdataloader:
    predict = best_model(xx)
    stacked = torch.stack((predict.argmax(1), yy.argmax(1)), 1)
    cmt = torch.zeros(lablesize, lablesize, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
cmt = cmt.numpy()
for i in range(lablesize):
    print('lable#{}_acc = '.format(i+1), cmt[i,i] / np.sum(cmt[:,i]) * 100)


torch.save(best_model, 'model.pth')
np.savetxt('history.txt', np.vstack([train_losses, test_losses, correct_hist]).T)
np.savetxt('conf_matrix.txt', cmt)

plot = PlotClass()
for i in range(lablesize):
    plot.tvsp(predict.detach().cpu().numpy()[:, i], yy.detach().cpu().numpy()[:, i])
plot.loss(test_losses, train_losses)

plt.figure(300)
plt.grid()
plt.plot(correct_hist, marker='None', color='b', linestyle='-', linewidth=1, alpha=1, label='Training Set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('./result_images/Acc.jpg')

ttttt = (np.sum(np.diagonal(cmt, 1)) + np.sum(np.diagonal(cmt, 0)) + np.sum(np.diagonal(cmt, -1))) / np.sum(cmt)
