
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from models.utils.continual_model import ContinualModel
from utils.per_buffer import PERBuffer
from utils.prompt_pool import PromptPool
from utils.args import *
import math
transformers.logging.set_verbosity(50)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='dualp_vit')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--pw',type=float,default=0.5,help='Penalty weight.')
    parser.add_argument('--freeze_clf',type=int,default=0,help='clf freeze flag')
    parser.add_argument('--init_type',type=str,default='default',help='prompt & key initialization')

    return parser

class DUALPVIT(ContinualModel):
    NAME = 'dualp_vit'
    COMPATIBILITY = ['class-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(DUALPVIT, self).__init__(backbone, loss, args, transform)
        
        self.vitEmbeddings = self.net.vit.embeddings
        self.vitEncoder = self.net.vit.encoder 
        self.layernorm = self.net.vit.layernorm
        self.classifier = self.net.classifier
        
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

        self.lr = args.lr
        self.args = args

        # ! cifar100 L2P official param setting
        self.topN = 1
        self.prompt_num = 20  # ! equal to 'expert prompt length' in dual prompt paper
        self.gprompt_num = 4 # ! equal to 'general prompt length' in duap prompt paper
        self.pool_size = 10  # ! pool size per layer. if dual prompt : entire pool size = task num
        self.layer_g = [0,1] # layer that attach general prompt : first, second layer by paper
        self.layer_e = [2,3,4] # layer that attach expert prompt : third to fifth layer by paper
        
        # ! init promptpool
        self.pool = PromptPool()
        self.pool.initPool(12,self.pool_size,self.prompt_num,768,768,self.device,embedding_layer=None,init_type=args.init_type)
        self.general_prompt = [torch.rand((self.gprompt_num,768),requires_grad=True,device=self.device) for i in range(12)] #Lg = 5 in 

        self.init_opt(args)
    
    def init_opt(self,args):
        key_list = [e for layer_k in self.pool.key_list for e in layer_k]
        prompt_list = [e for layer_p in self.pool.prompt_list for e in layer_p]
        if args.freeze_clf == 0:
            self.learning_param = key_list+prompt_list+list(self.classifier.parameters())+self.general_prompt
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)
        else:
            self.learning_param = key_list+prompt_list+self.general_prompt
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)

    # ! untouched
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
        
    def getPrompts(self,layer,pool,keys):
        B = keys.shape[0]

        if layer in self.layer_g:
            prompts = self.general_prompt[layer].unsqueeze(0).repeat(B,1,1)
            return prompts
        
        elif layer in self.layer_e:
            pTensor = torch.stack(pool.prompt_list[layer])
            T, Lp, Dp = pTensor.shape

            # ! selectedKeys: (B, topN)
            prompts = pTensor[keys,:,:]  # ! (B, topN, Lp, Dp)
            prompts = prompts.reshape(B,-1,Dp)   # ! (B, topN*Lp, Dp)
            return prompts
        
        else:
            return None
    
    def selfAttention(self,layer, x, prompts, head_mask=None, output_attentions=False):
        selfatt_part = self.vitEncoder.layer[layer].attention.attention

        mixed_query_layer = selfatt_part.query(x)
        if prompts != None:
            half = int(prompts.shape[1]/2)
            key_layer = selfatt_part.transpose_for_scores(selfatt_part.key(torch.cat([x[:,0,:].unsqueeze(1), prompts[:, :half ,:] , x[:,1:,:]], 1)))
            value_layer = selfatt_part.transpose_for_scores(selfatt_part.value(torch.cat([x[:,0,:].unsqueeze(1), prompts[:, half: ,:] , x[:,1:,:]], 1)))
        else:
            key_layer = selfatt_part.transpose_for_scores(selfatt_part.key(x))
            value_layer = selfatt_part.transpose_for_scores(selfatt_part.value(x))
        
        query_layer = selfatt_part.transpose_for_scores(mixed_query_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(selfatt_part.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = selfatt_part.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (selfatt_part.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
        
    
    def vitAttention(self,layer,x,prompts,head_mask=None, output_attentions=False):
        attention_part = self.vitEncoder.layer[layer].attention
        self_outputs = self.selfAttention(layer, x, prompts, head_mask, output_attentions)

        attention_output = attention_part.output(self_outputs[0], x)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    
    def encoder_block(self,layer,x,prompts,head_mask=None, output_attentions=False):
        # code from huggingface ViTLayer
        block = self.vitEncoder.layer[layer]
        self_attention_outputs = self.vitAttention(
            layer,
            block.layernorm_before(x),  # in ViT, layernorm is applied before self-attention
            prompts,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        x = attention_output + x

        # in ViT, layernorm is also applied after self-attention
        layer_output = block.layernorm_after(x)

        layer_output = block.intermediate(layer_output)

        # second residual connection is done here
        layer_output = block.output(layer_output, x)

        outputs = (layer_output,) + outputs

        return outputs
    
    def vitlayers(self,x,keys):
        for i in range(len(self.vitEncoder.layer)):
            if i < self.pool.layer:
                prompts = self.getPrompts(i,self.pool,keys)
            x = self.encoder_block(i,x,prompts)[0]

        x = self.layernorm(x)
        z_clf = x[:, 0, :] # 논문에 정확한 방법이 안나와 있어서 임의로 cls토큰 사용
        output = self.classifier(z_clf)

        return output, z_clf

    def forward_dualp(self, inputs, task_id=None):
        embedding = self.vitEmbeddings(inputs) # B x cls+L x dim
        representations = self.vitEncoder(embedding[:,1:,:])[0] # make query with out [cls] token
        # ! make query with [CLS] token representation ... (B, 768)
        query = representations[:, 0, :]

        kTensor = torch.stack(self.pool.key_list[0])

        if (self.net.training == False):
            distance, keys = self.similarity(self.pool,query,kTensor,self.topN)
        else:
            keys = torch.tensor(task_id,requires_grad=False,device=self.device).unsqueeze(0).repeat(embedding.shape[0],1)
            q = nn.functional.normalize(query,dim=-1)
            k = nn.functional.normalize(kTensor,dim=-1)
            distance = 1-torch.matmul(q,k.T)[:,task_id].unsqueeze(1)
        
        logits, z_clf = self.vitlayers(embedding,keys)
        
        return logits, distance , z_clf

    # ! util functions for compatibility with other experiment pipeline
    def forward_model(self, x: torch.Tensor, task_id=None):
        if self.pool == None:
            return self.net(x, task_id=task_id)
        logits, distance, z_clf = self.forward_dualp(x, task_id)
        
        return logits
        
    def observe(self, inputs, labels,dataset,t):
        logits, distance, z_clf  = self.forward_dualp(inputs,t)
        logits[:, 0:t * dataset.N_CLASSES_PER_TASK] = -float('inf')
        logits[:, (t + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf')
        loss = self.loss(logits, labels) + self.args.pw*torch.mean(torch.sum(distance,dim=1))
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()