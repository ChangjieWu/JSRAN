import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from densenet import DenseNet

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], reduction=params[' reduction'],
                                bottleneck=params['bottleneck'], use_dropout=params['use_dropout'])
        self.init_hidden = nn.Linear(params['D'], params['n'])

    def forward(self, params, x):
        ctx = self.encoder(x)  
        area = ctx.shape[2]*ctx.shape[3]
        ctx_mean = ctx.sum(3).sum(2) / area  
        init_state = torch.tanh(self.init_hidden(ctx_mean)) 
        return ctx, init_state

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.emb_model = nn.Embedding(params['K']+1, params['m'])

        self.gru_1 = nn.GRUCell(params['m'],params['n'])

        self.conv_Ua = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_Wa = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_Q = nn.Conv2d(1, params['M'], kernel_size=3, bias=False, padding=1)
        self.fc_Uf = nn.Linear(params['M'], params['dim_attention'])
        self.fc_va = nn.Linear(params['dim_attention'], 1)

        self.gru_2 = nn.GRUCell(params['D'],params['n'])

        self.fc_Wct = nn.Linear(params['D'], params['m'])
        self.fc_Wht = nn.Linear(params['n'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0 = nn.Linear(int(params['m'] / 2), params['K'])
    def forward(self, params, y, y_mask=None, context=None, pre_hidden=None, alpha_past=None):
    
        embedded = self.emb_model(y) 
        
        s_hat = self.gru_1(embedded,pre_hidden) 
        if y_mask is not None:
            s_hat = y_mask[:, None] * s_hat + (1 - y_mask)[:, None] * pre_hidden
        
        Wa_h1 = self.fc_Wa(s_hat)  

        Ua_ctx = self.conv_Ua(context)  
        Ua_ctx = Ua_ctx.permute(0, 2, 3, 1) 
        alpha_past_ = alpha_past[:, None, :, :] 
        cover_F = self.conv_Q(alpha_past_) 
        cover_F = cover_F.permute(0, 2, 3, 1)  
        cover_vector = self.fc_Uf(cover_F)  
        e_attention = Ua_ctx + Wa_h1[:, None, None, :] + cover_vector  
        e_attention = torch.tanh(e_attention)

        alpha = self.fc_va(e_attention)  
        shape_alpha = alpha.shape
        alpha = alpha.view(alpha.shape[0], -1) 
        alpha = F.softmax(alpha,1)
        ct = context.view(context.shape[0],context.shape[1],-1)*alpha[:,None,:]
        ct = ct.sum(2)
        
        s = self.gru_2(ct,s_hat) 
        if y_mask is not None:
            s = y_mask[:, None] * s + (1 - y_mask)[:, None] * s_hat

        logit = self.fc_Wct(ct) + self.fc_Wht(s) + embedded 
        logit = logit.view(logit.shape[0],-1,2) 
        logit = logit.max(2)[0] 
        if params['use_dropout']:
            logit = self.dropout(logit)
        out = self.fc_W0(logit)  
        alpha = alpha.view(shape_alpha[:3])
        return out, s, alpha