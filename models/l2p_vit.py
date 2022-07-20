import torch
import torch.nn as nn
import transformers
from models.utils.continual_model import ContinualModel
from utils.per_buffer import PERBuffer
from utils.prompt_pool import PromptPool
from utils.args import *
transformers.logging.set_verbosity(50)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='l2p_vit')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--pw',type=float,default=0.5,help='Penalty weight.')
    parser.add_argument('--freeze_clf',type=int,default=0,help='clf freeze flag')
    parser.add_argument('--init_type',type=str,default='default',help='prompt & key initialization')

    return parser

class L2PVIT(ContinualModel):
    NAME = 'l2p_vit'
    COMPATIBILITY = ['class-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(L2PVIT, self).__init__(backbone, loss, args, transform)
        
        self.vitEmbeddings = self.net.vit.embeddings
        self.vitEncoder = self.net.vit.encoder 
        self.layernorm = self.net.vit.layernorm
        self.classifier = self.net.classifier
        
        #'freeze_part': ['encoder', 'embedding', 'cls']
        self.net.requires_grad_(False)
        self.vitEmbeddings.requires_grad_(False)
        self.vitEncoder.requires_grad_(False)
        self.layernorm.requires_grad_(False)
        
        if args.freeze_clf == 0: 
            self.classifier.requires_grad_(True)
            #torch.nn.init.zeros_(self.classifier.weight)
        else:                    
            self.classifier.requires_grad_(False)

        self.learning_param = None

        self.args = args
        self.lr = args.lr

        # ! cifar100 L2P official param setting
        self.topN = 5
        self.prompt_num = 5  # ! equal to 'prompt length' in l2p paper
        self.pool_size = 10  # ! pool size per layer. if l2p : entire pool size
        
        # ! init promptpool
        self.pool = PromptPool()
        self.pool.initPool(1,self.pool_size,self.prompt_num,768,768,self.device,embedding_layer=None,init_type=args.init_type)
        self.init_opt(args)
        #self.buffer = PERBuffer(self.args.buffer_size, self.device)
    
    def init_opt(self,args):
        self.pool.key_freq_past = self.pool.key_freq_now.clone().detach()
        key_list = [e for layer_k in self.pool.key_list for e in layer_k]
        prompt_list = [e for layer_p in self.pool.prompt_list for e in layer_p]
        if args.freeze_clf == 0:
            self.learning_param = key_list+prompt_list+list(self.classifier.parameters())
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)
        else:
            self.learning_param = key_list+prompt_list
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)

    def similarity(self, pool, q, k, topN):
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        sim = torch.matmul(q, k.T)  # (B, T)
        dist = 1 - sim

        val, idx = torch.topk(dist, topN, dim=1, largest=False)

        # topk에 해당하는 distance만 모으는 과정(gather함수는 reproducibility불가)
        dist_pick = []
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))

        dist = torch.stack(dist_pick)

        return dist, idx
        
    def getPrompts(self,pool,query):
        B, D = query.shape
        pTensor = torch.stack(pool.prompt_list[0])
        kTensor = torch.stack(pool.key_list[0])
        T, Lp, Dp = pTensor.shape
        T, D = kTensor.shape

        # ! selectedKeys: (B, topN)
        distance, selectedKeys = self.similarity(pool,query,kTensor,self.topN) 
        prompts = pTensor[selectedKeys,:,:]  # ! (B, topN, Lp, Dp)
        prompts = prompts.reshape(B,-1,Dp)   # ! (B, topN*Lp, Dp)
        return prompts, distance, selectedKeys

    def vitlayers(self,x,prompt_length,boundary=None):
        # ! [B, 0:N*Lp, D] -> POOLING -> [B, D] (prompt part except [cls] token)
        z_prompted = self.layernorm(self.vitEncoder(x)[0])[:, 1:prompt_length+1, :]
        z_clf = torch.mean(z_prompted, dim=1)
        return self.classifier(z_clf), z_clf

    def forward_l2p(self, inputs, task_id=None):
        embedding = self.vitEmbeddings(inputs) # B x cls+L x dim
        representations = self.vitEncoder(embedding[:,1:,:])[0] # make query with out [cls] token

        # ! make query with [CLS] token representation ... (B, 768)
        query = representations[:, 0, :]
        
        # ! get prompts(B, topN*pnum, 768) those are most similar with query
        prompts, distance, selectedKeys = self.getPrompts(self.pool, query)
        B, NLp, Dp = prompts.shape
        prompted_x = torch.cat([embedding[:,0,:].unsqueeze(1), prompts, embedding[:,1:,:]], 1) # [cls]+prefix+input
        #print(prompted_x.shape) # length = cls + top5 x 5token + input =  1 + (5x5) + 194
        logits, z_clf = self.vitlayers(prompted_x, prompt_length=NLp)
        
        return logits, distance, z_clf

    # ! util functions for compatibility with other experiment pipeline
    def forward_model(self, x: torch.Tensor, task_id=None):
        if self.pool == None:
            return self.net(x, task_id=task_id)
        logits, distance, z_clf = self.forward_l2p(x, task_id)
        
        return logits
        
    def observe(self, inputs, labels,dataset,t):
        logits, distance, z_clf  = self.forward_l2p(inputs)
        logits_original = logits.clone().detach()
        logits[:, 0:t * dataset.N_CLASSES_PER_TASK] = -float('inf')
        logits[:, (t + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf')
        loss = self.loss(logits, labels) + self.args.pw*torch.mean(torch.sum(distance,dim=1))
        
        """ # ! ============================= use replay buffer
        if not self.buffer.is_empty():
            # begin: Loss 2
            (m_examples, m_labels, m_features), choice1 = self.buffer.get_data(self.args.minibatch_size)
            buf_outputs, distance, z_clf  = self.forward_l2p(m_examples)
            loss += self.args.beta * self.loss(buf_outputs, m_labels)
            # end: Loss 2
        
            # begin: Loss 3
            (m_examples, m_labels, m_features), choice1 = self.buffer.get_data(self.args.minibatch_size)
            buf_feat_gen, distance, z_clf  = self.forward_l2p(m_examples)
            loss += self.args.alpha * F.mse_loss(buf_feat_gen, m_features)
            # end: Loss 3   
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        self.buffer.add_data(examples=inputs, labels=labels, features=logits_original.data)
        # ! ============================= """
        
        #======================= loss if not use replay buffer
        self.opt.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.learning_param, 1.0)
        self.opt.step()
        #=======================

        return loss.item()