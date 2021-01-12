# __author__ = 'lijiaran'
# coding=UTF-8
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import utils
from utils import encoderRNN, MLP
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# word_list, encoded_labels, n_worker, encoded_cls, pad_sens_tensor,int_to_word = utils.load_movie_review()
# crowd_data_mask = utils.crowd_data_mask(encoded_cls)
# print(f"word_size:\t{len(word_list)}")
word_list, encoded_labels, n_worker, crowd_data_mask, pad_sens_tensor,int_to_word = utils.load_tweets10k()
# hidden, encoded_labels, crowd_data_mask, n_worker = utils.load_blue_birds()
# hidden, encoded_labels, crowd_data_mask, n_worker = utils.load_MS()

embedding_matrix = utils.build_embedding_matrix(int_to_word, word_dim=200)
word_size = 200
hidden_size = 10
n_label = crowd_data_mask.size(2)
#-------------------------------------------------------   PCA   -------------------------------------------------------
# U_mat, S_mat, V_mat = torch.pca_lowrank(hidden)
# encoded_reviews = torch.matmul(hidden, V_mat)
# print(encoded_reviews)
#------------------------------------------------mlp encoder------------------------------------------------------------
# with torch.no_grad():
#     mlp = MLP(hidden.size(1 ),hidden_size)
#     encoded_reviews = mlp(hidden.float())
# print(encoded_reviews)
#---------------------------------- --------------AutoEncoder-----------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.BatchNorm1d(input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size//4),
            nn.BatchNorm1d(input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, hidden_size),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size//4),
            nn.BatchNorm1d(input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, input_size//2),
            nn.BatchNorm1d(input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def cross_entropy_loss(input, target):
    loss = -torch.mean(torch.sum(input.mul(torch.log(1e-10+target)+(1.0-input).mul(torch.log(1e-10+(1.0-target)))),-1))
    return loss

# auto_encoder = AutoEncoder(hidden.size(1), hidden_size)
# optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=0.02, weight_decay=1e-6)
# criterion = nn.MSELoss()
# print("pretrain encoded reviews!")
# for epoch in range(2000):
#     encoded, decoded = auto_encoder(hidden)
#     loss = criterion(hidden, decoded)
#     optimizer.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     if epoch % 50 == 0:
#         print('epoch [{}/{}], loss:{:.8f}'.format(epoch+50, 2000, loss))
# encoded_reviews=encoded.detach()
#-----------------------------------------------------------------------------------------------------------------------
encoder_rnn = encoderRNN(len(word_list), hidden_size, word_size, torch.from_numpy(embedding_matrix))
encoder_rnn.to(device)
output, hidden = encoder_rnn(pad_sens_tensor)
encoded_reviews = hidden.squeeze()
encoded_labels = torch.LongTensor([label for idx, label in enumerate(encoded_labels) if len(encoded_reviews[idx]) > 0] )


batch_size = 20
shuffle_index = [i for i in range(len(encoded_reviews))]
random.shuffle(shuffle_index)
with open('index.txt','a') as f:
    f.write(str(shuffle_index) + '\n')

def para_save(para, fname):
    data_file = os.path.join('dats','movie_review',fname)
    if os.path.exists(data_file):
        print(f"loading para:{data_file}")
        data = pickle.load(open(data_file, 'rb'))
    else:
        print(f"{data_file} created!")
        data = para
        pickle.dump(data, open(data_file, 'wb'))
    return data
# train_x = para_save(encoded_reviews[shuffle_index], 'encoded_reviews.dat')
# train_y = para_save(encoded_labels[shuffle_index], 'encoded_labels.dat')
# train_m = para_save(crowd_data_mask[shuffle_index],'crowd_data_mask.dat')
# train_x = train_x.to(device)
# train_y = train_y.to(device)
# train_m = train_m.to(device)
train_x = encoded_reviews[shuffle_index].to(device)
train_y = encoded_labels[shuffle_index].to(device)
train_m = crowd_data_mask[shuffle_index].to(device)


def majority_vote_continuous(crowd_labels):   # N * n_worker * n_labels
    final_vote = torch.sum(crowd_labels,1)
    soft = torch.nn.Softmax(dim = 1)
    final_label = soft(final_vote)
    return final_label


def majority_vote(crowd_labels):  # N * n_worker * n_labels
    final_label = torch.zeros(crowd_labels.size(0), crowd_labels.size(2))
    final_vote = torch.sum(crowd_labels,1)
    for i in range(len(crowd_labels)):
        temp = torch.argmax(final_vote, 1)
        final_label[i, temp[i].item()] = 1.0
    return final_label

class DS():

    def __init__(self, crowd_data_mask, encoded_labels, ds_em = 5):
        self.ds_em = ds_em
        self.encoded_labels = encoded_labels
        self.crowd_labels = crowd_data_mask
        self.n_task = crowd_data_mask.size(0)
        self.n_worker = crowd_data_mask.size(1)
        self.n_label = crowd_data_mask.size(2)
        self.soft = torch.nn.Softmax(dim=1)

        self.pi = torch.ones(self.n_worker, self.n_label, self.n_label)
        self.gt = torch.zeros(self.n_task, self.n_label)
        # initialize estimated ground truth with majority voting
#         self.gt = majority_vote(self.crowd_labels)  # N * n_label  (discrete label)
        self.gt = majority_vote_continuous(self.crowd_labels)  # N * n_label  (distribution)

    def e_step(self):
        print("E-step")
        for i in range(self.n_task):
            temp = 1.0* torch.ones(self.n_label)
#             temp = torch.zeros(self.n_label)
            for idx, r in enumerate(torch.sum(self.crowd_labels[i], 1)):
                if r == 1:
#                     temp *= self.pi[idx, :, torch.argmax(self.crowd_labels[i,idx,:]).item()]
                    temp += torch.log(self.pi[idx, :, torch.argmax(self.crowd_labels[i,idx,:]).item()])
#             print(temp)
            self.gt[i,:] = self.gt[i,:] * torch.exp(temp)
        return self.gt

    def m_step(self):
        print("M-step")
        for r in range(self.n_worker):
            normalizer = torch.zeros(self.n_label)
            for i, sen in enumerate(torch.sum(self.crowd_labels[:,r,:],1)):
                if sen == 1:
                    self.pi[r,:,torch.argmax(self.crowd_labels[i,r,:])] += self.gt[i,:]
                    normalizer += self.gt[i,:]
            self.pi[r,:,:] = self.pi[r,:,:] / normalizer.unsqueeze(1)
        return self.pi

    def eval(self):
        self.final_label = torch.argmax(self.gt, 1)
        self.correct_num = sum(self.final_label.eq(self.encoded_labels))
        self.acc = 1.0*self.correct_num/len(self.encoded_labels)
        print("iter_number: {}\tDS Accuracy:\t{:.4f}%".format(self.ds_em, 100. *self.acc.item()))

    def em(self):
        self.m_step()
        for i in range(self.ds_em):
            self.e_step()
            self.m_step()
        self.gt = F.softmax(self.gt, 1)
        self.pi = F.softmax(self.pi, 2)
        self.eval()


class Transform(nn.Module):
    def __init__(self, worker_abi):
        super(Transform, self).__init__()
        self.worker_abi = worker_abi.double()    # initialized by DS
#         self.worker_abi = torch.nn.Parameter(torch.randn(n_worker, n_label,n_label))   # random initialization
        self.softmax = nn.Softmax(dim=1)
        self.softmax2  = nn.Softmax(dim=2)

    def forward(self, f_score):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.worker_abi.to(device)
        temp_abi= self.softmax2(self.worker_abi)
        w_abi = torch.unbind(temp_abi, 0)
        ob_labels = torch.FloatTensor([]).to(device)
        for each in w_abi:
            temp = torch.matmul(f_score, each)
            ob_labels = torch.cat((ob_labels, temp.unsqueeze(2)), 2)
        ob_labels_soft = self.softmax(ob_labels)                         # b * n_label * n_worker
#
        return ob_labels, ob_labels_soft


class Test_method(nn.Module):
    def __init__(self, n_label, n_worker, hidden_size, worker_abi):
        super(Test_method, self).__init__()
        self.n_worker = n_worker
        self.n_label = n_label
        self.lab_embedding = nn.Embedding(n_label, hidden_size)
        self.worker_embedding = nn.Embedding(n_worker, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.transform = Transform(worker_abi)

    def forward(self, hidden_sen, crowd_mask):
        worker_embedded = self.worker_embedding.weight
        crowd_s_worker = torch.sum(crowd_mask, 2)            # batch_size * n_worker
        crowd_s_worker = crowd_s_worker.squeeze()            # batch_size * n_worker
#         print(hidden_sen.shape)
#         print(worker_embedded.t().shape)
        att_weight = torch.matmul(hidden_sen, worker_embedded.t()).mul(crowd_s_worker) # batch_size * n_worker
        for i in range(len(att_weight)):
            for j in range(n_worker) :
                if att_weight[i,j] == 0:
                    att_weight[i,j] = -99999999
                    # print(worker)
        att_weight = self.softmax(att_weight)
#         print(len(torch.nonzero(att_weight)))
        f_score = torch.matmul(att_weight.unsqueeze(1), crowd_mask)   # batch_size * 1 * n_label
        ob_labels, ob_labels_soft= self.transform(f_score.squeeze())
        return f_score.squeeze(), ob_labels, ob_labels_soft, att_weight


#     def forward(self, hidden_sen, crowd_mask):
#
#         worker_embedded = self.worker_embedding.weight
#         lab_embedded = self.lab_embedding.weight
#
#         score_1 = torch.chain_matmul(hidden_sen, worker_embedded.t())
#         score_2 = torch.chain_matmul(hidden_sen, lab_embedded.t())
#         score_3 = torch.chain_matmul(worker_embedded, lab_embedded.t())
#
#         f_temp_score = torch.matmul(score_1.unsqueeze(2), score_2.unsqueeze(1)).mul(score_3.expand([hidden_sen.size(0),self.n_worker,self.n_label]))
#         f_temp_score = torch.sigmoid(f_temp_score)
#         f_score = f_temp_score.mul(crowd_mask)
#         f_score = torch.sum(f_score,1)
#         f_out = self.softmax(f_score)
#         return f_score, f_out
#-------------------------MV ini---------------------------------------
ini_mv_abi = torch.zeros(n_worker,n_label,n_label)
mv_true_label = majority_vote_continuous(crowd_data_mask)
mv_true_label = torch.argmax(mv_true_label, 1)
for i in range(len(mv_true_label)):
    for j in range(n_worker):
        if sum(crowd_data_mask[i,j])!=0:
            ini_mv_abi[j,mv_true_label[i],torch.argmax(crowd_data_mask[i,j,:])]+=1
#-------------------------ds ini---------------------------------------
dw = DS(crowd_data_mask, encoded_labels, ds_em=2)
dw.em()
# print(dw.pi[:20])
#-------------------------0.9-0.1 ini-------------------------------
smooth = 0.1
worker_abi_ini = (torch.eye(n_label)*(1-smooth)+smooth/n_label).repeat(n_worker,1,1)
#print(worker_abi_ini.shape)
#-----------------------------------------------------
tt = Test_method(n_label, n_worker, hidden_size, dw.pi)        ####
tt = tt.double()
tt.to(device)
print(tt)
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(tt.parameters(), lr=0.005)
print("Model's state_dict:")
for param_tensor in tt.state_dict():
    print(param_tensor, "\t", tt.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# -------------------------------------- 数据维度计算测试------------------------------------------------------------
# f_score, f_out, ob_labels, ob_labels_out= tt(train_x[0:20], train_m[0:20])
# # print(ob_labels_out)
# f_ob_labels = utils.crowd_cat(ob_labels_out)           # 2*(203*batch_size) 拼接
# print(f_ob_labels.shape)
# temp = torch.sum(crowd_data_mask, 2).squeeze()         # 4999 * 203
# crowd_target = torch.argmax(crowd_data_mask,dim=2)     # 4999 * 203
# print(crowd_data_mask[6])
# print(crowd_target.shape)
# ee = utils.crowd_cat(crowd_target[0:2],dim=-1)
# te = utils.crowd_cat(temp[0:2],dim=-1)
# print(te)
# print(crowd_target.mul(temp)[6])
#--------------------------------------------------------------------------------------------------------------------
tt.train()
best_acc = 0
for epoch in range(200):
    correct_num = 0
    final_labels = torch.Tensor([]).to(device)
    for batch_idx in range(math.ceil(len(train_x)/batch_size)):
        correct_batch = 0
        trained_batch = 0
        if batch_idx != int(len(train_x)/batch_size):
            batch_train_x = train_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch_train_y = train_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch_train_m = train_m[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            batch_train_x = train_x[batch_idx*batch_size:]
            batch_train_y = train_y[batch_idx*batch_size:]
            batch_train_m = train_m[batch_idx*batch_size:]
        optimizer.zero_grad()
        f_score, ob_labels, ob_labels_soft,att_weight = tt(batch_train_x.double(), batch_train_m.double())
#         print(ob_labels)
        f_ob_labels = utils.crowd_cat(ob_labels)
        mask_temp = torch.sum(batch_train_m, 2).squeeze()
        mask =  mask_temp.view(-1)
        crowd_target_temp = torch.argmax(batch_train_m,dim=2)
        crowd_target = crowd_target_temp.view(-1)
        loss = criterion(f_ob_labels.t(), crowd_target)
        loss = (sum(loss.mul(mask)))/sum(mask)
        loss.backward(retain_graph=True)
        optimizer.step()
        final_label =torch.argmax(f_score, dim=1).squeeze()
        final_labels = torch.cat((final_labels, final_label), 0)
        correct_tensor = final_label.eq(batch_train_y)
        correct_num += sum(1.0*correct_tensor)
        correct_batch += sum(1.0*correct_tensor)
        trained_batch += len(final_label)
        acc = 1.0*correct_batch/trained_batch
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.4f}'.format(epoch+1, batch_idx * batch_size, len(train_x),
                   100. * batch_idx  / (int(len(train_x)/batch_size)+1), loss.item(), acc.item()))
    correct_final = final_labels.eq(train_y)
    list_conrrect = correct_final.cpu().numpy().tolist()
    # print(list_conrrect)
    index_wrong_temp = [i for i,x in enumerate(list_conrrect) if x == 0]
    # print(index_wrong_temp)
    index_wrong = [shuffle_index[x] for x in index_wrong_temp ]
    # print(index_wrong)
    # print('Wrong Answer:\t', len(index_wrong))
    final_acc = 100. * correct_num/len(train_x)
    if final_acc.item() > best_acc:
        best_acc = final_acc.item()
        best_epoch = epoch+1
    with open('final_labels.txt','a') as f:
        f.write("epoch"+str(epoch+1)+'\n'+"best_epoch"+str(best_epoch)+'\n'+"acc:\t"+str(best_acc)+'\n'+str(index_wrong) + '\n')
    print("current_acc:\t{:.4f}%\nbest_epoch:\t{}\nbest_acc:\t{:.4f}%".format(final_acc.item(),best_epoch, best_acc))
#     print('attention weight:', att_weight)
    print("worker_ability:", tt.transform.softmax2(tt.transform.worker_abi[:20]))
#     t_score, t_out,t_ob,t_ob_soft = tt(test_x, test_m)
# #     loss = criterion(t_score, test_y)
#     final_t_label =torch.argmax(t_out, dim=1)
#     correct_t_tensor = final_t_label.eq(test_y)
#     acc = sum(1.0*correct_t_tensor)/len(final_t_label)
#     print('Test acc:\t {:.6f}'.format(acc.item()))