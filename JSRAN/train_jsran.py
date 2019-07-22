import time
import random
import os
import re
import sys
import copy
import pickle
import numpy as np
from PIL import Image
import torch 
from torch import optim, nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from JSRAN import Encoder,Decoder
import skimage.transform as trans
import warnings

warnings.filterwarnings('ignore')

def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon


def compute_acc(gt,predic):
    right = (gt == predic).sum(1) - gt.shape[1]
    num_right = (right>=0).sum(0) 
    return num_right

class Load_data(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary
        
    def __call__(self, batch):
        char2id = self.dictionary
        images, labels, rotations = zip(*batch)
        n_batch = len(images)

        imgs = []
        for i, img in enumerate(images):
            rotate = rotations[i]
            noise = random.uniform(-10.,10.) + rotate
            img = trans.rotate(img, noise, mode = 'edge')
            img = trans.resize(img,(48,48))
            img = torch.from_numpy(img)
            img = img.to(torch.float)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)

        dictionary = self.dictionary
        new_labels = []
        len_labels = []
        for label in labels:
            label2index = []
            len_labels.append(len(label))
            for w in label:
                if dictionary.__contains__(w):
                    label2index.append(dictionary[w])
                else:
                    print('a word not in the dictionary !!', w)
                    sys.exit()
            new_labels.append(label2index)
        max_length = max(len_labels)
        np_labels = np.zeros((max_length+1,n_batch)).astype(np.int64)
        np_labels_mask = np.zeros((max_length+1,n_batch)).astype(np.float32)
        for idx,label in enumerate(new_labels):
            np_labels[:len_labels[idx],idx] = label
            np_labels_mask[:len_labels[idx]+1,idx] = 1.
        new_labels = torch.from_numpy(np_labels)
        new_labels_mask = torch.from_numpy(np_labels_mask)
                
        return imgs, new_labels, new_labels_mask

class LoadPartDataset(Dataset):
    def __init__(self, img_path, label_path, data_range):
        load_time = time.time()
        self.imgs = []
        self.labels = []
        self.rotations = []
        with open(label_path, 'rb') as fp_label:
            labs = pickle.load(fp_label)

        for img_p in img_path:
            with open(img_p,'rb') as fp:
                font_imgs = pickle.load(fp)
                rotation = -60
                for t_range in data_range:
                    start_id = t_range[0]
                    end_id = t_range[1]
                    t_imgs = font_imgs[start_id:end_id]
                    t_labels = labs[start_id:end_id]
                    t_rotations = [rotation for i in range(len(t_imgs))]
                    rotation += 20
                    self.imgs.extend(t_imgs)
                    self.labels.extend(t_labels)
                    self.rotations.extend(t_rotations)
        load_time = (time.time() - load_time)/60
        print(f'Loading dataset: label:{len(self.labels)} img:{len(self.imgs)} rotate:{len(self.rotations)}')
        print(f'load time: {load_time:.2f} min, check rotation:{rotation}')
    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        rotate = self.rotations[index]
        return image, label, rotate

    def __len__(self):
        return len(self.labels)

torch.cuda.set_device(2)
device = torch.device("cuda:2")
pre_train = False
num_angles = 7
num_fonts = 20
num_angle_base = 20
class_K = 417
valid_batch_size = 100
fonts = [i for i in range(0,num_fonts)]
img_dataset = ['../data/img_pkl/'+str(i).zfill(3)+'_img_3755.pkl' for i in fonts]
train_range = [ [0,500],[0,500],[0,500],[0,3000],[0,500],[0,500],[0,500]]
test_range = [  [2900,3000],[2900,3000],[2900,3000],[2900,3000],[2900,3000],[2900,3000],[2900,3000] ]
caption_dataset = ['../data/cap_pkl/cap_3755.pkl']
dictionaries = ['../data/dictionary_v2.txt']
saveto = [r'./result/stn_wap_encoder_params.pkl',r'./result/stn_wap_decoder_params.pkl',
          r'./result/stn_wap_encoder_params_decay.pkl',r'./result/stn_wap_decoder_params_decay.pkl',
          r'./result/stn_wap_encoder_params_end.pkl',r'./result/stn_wap_decoder_params_end.pkl']
valid_output = [r'./result/valid_output.txt']

dispFreq = 200
valid_freq = 9500
lrate = 0.1
pre_epoch = 0
estop = False
halfLrFlag = 0
bad_counter = 0
pre_best_acc = 0
history_acc = []
patience = 15




max_epochs = 5000
batch_size = 100
decay_c = 1e-4
clip_c = 100.

params = {}
params['K'] = class_K
params['n'] = 256
params['m'] = 256
params['M'] = 512
params['dim_attention'] = 512
params['D'] = 936
params['growthRate'] = 24
params[' reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 3


worddicts = load_dict(dictionaries[0])  
worddicts_r = [None] * len(worddicts)  
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk
print('Prepare dictionaries')

print('Load data')
load_data = Load_data(worddicts)
train_dataset = LoadPartDataset(img_path=img_dataset, label_path=caption_dataset[0], data_range=train_range)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=load_data)
test_dataset = LoadPartDataset(img_path=img_dataset, label_path=caption_dataset[0], data_range=test_range)
valid_loader = DataLoader(dataset=test_dataset, batch_size=valid_batch_size, num_workers=8, collate_fn=load_data)


print('initialize model')
decoder = Decoder(params)
encoder = Encoder(params)
print('Net to GPU')
decoder.to(device)
encoder.to(device)


print('load pre_train model')
if pre_train:
    pretrained_dict = torch.load(saveto[0], map_location=device)
    encoder.load_state_dict(pretrained_dict)
    pretrained_dict = torch.load(saveto[1], map_location=device)
    decoder.load_state_dict(pretrained_dict)


print('set loss function and optimizer function')
criterion = torch.nn.CrossEntropyLoss(reduce=False)  # (y_maxlen.batch,)
encoder_optimizer = optim.Adadelta(encoder.parameters(),lr=lrate, weight_decay=decay_c)
decoder_optimizer = optim.Adadelta(decoder.parameters(),lr=lrate, weight_decay=decay_c)

ud_s = 0
print(f'begin tarining')
for eidx in range(max_epochs):
    
    n_samples = 0 
    n_batches = 0
    loss_s = 0.
    loss_epoch = 0.
    ud_epoch = time.time()
    
    
    for x, y, y_mask in train_loader:
        ud_start = time.time()
        loss = 0.
        n_samples += len(x)  
        n_batches += 1 
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        #----------------encode-----------------------------------
        encoder.train()
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        ctx, init_state = encoder(params, x)
        #----------------decode------------------------------------
        decoder.train()
        y_input = torch.ones(y.shape[1],dtype=torch.long,device=device)*params['K']
        s = init_state
        alpha_past = torch.zeros(ctx.shape[0], ctx.shape[2], ctx.shape[3], dtype=torch.float32,device=device)
        alphas = []
        for ei in range(y.shape[0]):
            y_mask_input = y_mask[ei]
            score, s, alpha = decoder(params, y_input, y_mask_input, ctx, s, alpha_past)
            alpha_past = alpha + alpha_past
            y_input = y[ei]
            alphas.append(alpha.cpu().detach().numpy())
            loss = criterion(score,y_input)*y_mask_input + loss
        
        loss = loss/(y_mask.sum(0))
        loss = loss.mean()
        
        if np.isnan(loss.item()) or np.isinf(loss.item()):
            print('NaN detected')
            
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_c)
        encoder_optimizer.step()
        decoder_optimizer.step()

        ud = time.time() - ud_start
        ud_s += ud
        loss_s += loss.item()
        if n_batches%dispFreq==0:
            ud_s /= 60.
            loss_s /= dispFreq
            print(f'Epoch {eidx+pre_epoch} Update {n_batches} Cost {loss_s:.6f} UD {ud_s:.2f} lrate {lrate} bad_counter {bad_counter}')
            ud_s = 0
            loss_epoch += loss_s
            loss_s = 0.

    #----------------------------valid-----------------
    f_valid = open(valid_output[0], 'w')
    fi = 0
    angle_right = []
    valid_time = time.time()
    encoder.eval()
    decoder.eval()
    num_right = 0
    with torch.no_grad():
        for x, y, y_mask in valid_loader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            ctx, s = encoder(params, x)
            y_input = torch.zeros(y.shape[1], dtype=torch.long, device=device) + params['K']
            alpha_past = torch.zeros(ctx.shape[0], ctx.shape[2], ctx.shape[3], dtype=torch.float32,device=device)
            
            out_labels = torch.LongTensor(y.shape[1],1).to(device)
            for ei in range(y.shape[0]):
                y_mask_input = y_mask[ei]
                score, s, alpha = decoder(params, y_input, y_mask_input, ctx, s, alpha_past)
                alpha_past = alpha + alpha_past
                prob_y =F.softmax(score,1)
                
                max_logs, max_ys = prob_y.topk(1)
                y_input = max_ys.squeeze(1)

                out_labels = torch.cat((out_labels,max_ys),1)
            out_labels = out_labels[:,1:] 

            for i in range(out_labels.shape[0]):
                f_valid.write(str(fi))
                fi += 1
                for j in range(out_labels.shape[1]):
                    t_index = out_labels[i][j].item()
                    if t_index == 0:   
                        break
                    f_valid.write(' ' + worddicts_r[t_index])
                f_valid.write('\n')
            
            out_labels = out_labels * y_mask.transpose(0,1).long() #()
            num_right_t = compute_acc(y.transpose(0,1), out_labels)
            angle_right.append(num_right_t.item())
            num_right += num_right_t
    f_valid.close()
    valid_acc = float(num_right)/len(test_dataset)
    history_acc.append(valid_acc)  
    if valid_acc >= np.array(history_acc).max():
        bad_counter = 0
        print('Saving model params ... ')
        torch.save(encoder.state_dict(),saveto[0])
        torch.save(decoder.state_dict(),saveto[1])

    for angle in range(num_angles):
        num_angle_right = 0
        acc = 0
        for font in range(num_fonts):
            font_angle = font*num_angles + angle
            num_angle_right += angle_right[font_angle]
        acc = num_angle_right/num_angle_base
        print(f'{acc:.2f}%',end = ' ')
    print()

    if valid_acc < np.array(history_acc).max():
        bad_counter += 1
        if bad_counter > patience:
            if halfLrFlag == 3:
                print('Early Stop!Saving model params')
                torch.save(encoder.state_dict(),saveto[4])
                torch.save(decoder.state_dict(),saveto[5])
                estop = True
            else:
                print('Learning rate decay!Saving model params')
                torch.save(encoder.state_dict(),saveto[2])
                torch.save(decoder.state_dict(),saveto[3])
                bad_counter = 0
                lrate = lrate / 10.
                # change learning rate
                for param_group in encoder_optimizer.param_groups:
                    param_group['lr'] = lrate
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = lrate
                halfLrFlag += 1

    valid_time = (time.time()-valid_time) / 60.
    print('valid on ', len(test_dataset), end=' ')
    print('Valid Tiem: %.2f'%(valid_time), end=' ')
    print('ExpRate: %.2f%%' % (valid_acc*100))
    
    if estop:
        break

