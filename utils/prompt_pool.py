import pickle
import torch
import transformers
import copy
import torch_scatter as ts

transformers.logging.set_verbosity(50)


class PromptPool():
  def __init__(self):
    self.total = None #한 층에 해당하는 pool당 key 개수
    self.new = None #한 층에 새로 추가되는 key개수
    self.pnum = None #key하나에 달려있는 prompt 개수
    self.pdim = None #prompt dimension
    self.kdim = None #key dimension
    self.key_list = None #key를 저장하는 list
    self.prompt_list = None #prompt를 저장하는 list
    self.layer = None #prompt pool의 층 수
    self.taskID_dict = {} #TIL일 때 task id가 주어지면 바로 prompt정보를 줄 수 있게 하는 dictionary
    self.key_freq_past = None
    self.key_freq_now = None
    
  # ! support uniform init (jax default => U(0, 0.01))
  def initPool(self,layer,total,pnum,pdim,kdim,device,embedding_layer=None,init_type='default'):
    self.layer = layer
    self.total = total #한 층에 해당하는 pool당 key 개수
    self.new = 0
    self.pnum = pnum
    self.pdim = pdim
    self.kdim = kdim
    self.key_freq_past = torch.ones(total,requires_grad=False,device=device)
    self.key_freq_now = torch.ones(total,requires_grad=False,device=device)
    self.init_type = init_type

    if embedding_layer != None: # initialize key token with word embedding
      embedding_layer.to(device)
      self.key_list = []
      for i in range(self.layer):
        layer_pool = []

        for j in range(total):
          words = torch.randint(low=500,high=10000,size=(1,1),device=device)
          key = embedding_layer(words).squeeze().clone().detach()
          layer_pool.append(key.requires_grad_(True))

        self.key_list.append(layer_pool)
      
    else:
      self.key_list = []
      for i in range(self.layer):
        if self.init_type == 'default':
          self.key_list.append([torch.randn(kdim,requires_grad=True,device=device) for j in range(total)])
        elif self.init_type == 'unif':  # ! same with l2p initialization
          self.key_list.append([torch.tensor(0.01*torch.rand(kdim),requires_grad=True,device=device) for j in range(total)])
        else:
          raise ValueError('not supported init type')

    if embedding_layer != None: # initialize prompt token with word embedding
      embedding_layer.to(device)
      self.prompt_list = []
      for i in range(self.layer):
        layer_pool = []

        for j in range(total):
          words = torch.randint(low=500,high=10000,size=(1,pnum),device=device)
          prompts = embedding_layer(words).squeeze().clone().detach()
          layer_pool.append(prompts.requires_grad_(True))
        self.prompt_list.append(layer_pool)
    
    else:
      self.prompt_list = []
      for i in range(self.layer):
        if self.init_type == 'default':
          self.prompt_list.append([torch.randn((pnum,pdim),requires_grad=True,device=device) for j in range(total)])
        elif self.init_type == 'unif':  # ! same with l2p initialization
          self.prompt_list.append([torch.tensor(0.01*torch.rand((pnum,pdim)),requires_grad=True,device=device) for j in range(total)])
        else:
          raise ValueError('not supported init type')
    
    self.taskID_dict[len(self.taskID_dict.keys())] = self.total


  def freezePool(self):
    for l in range(len(self.key_list)):
      for i in range(len(self.key_list[l])):
        self.key_list[l][i] = self.key_list[l][i].clone().detach().requires_grad_(False)
    
    for l in range(len(self.prompt_list)):
      for i in range(len(self.prompt_list[l])):
        self.prompt_list[l][i] = self.prompt_list[l][i].clone().detach().requires_grad_(False)

  def loadPool(self,path=None,update_key=False,update_prompt=False,device='cpu'):
    with open(path,'rb') as f:
      pool = pickle.load(f)
      self.key_list = pool.key_list
      self.prompt_list = pool.prompt_list
      self.total = pool.total
      self.new = 0
      self.pnum = pool.pnum
      self.pdim = pool.pdim
      self.kdim = pool.kdim
      self.layer = pool.layer

      self.taskID_dict = pool.taskID_dict
    
    for l in range(len(self.key_list)):
      for i in range(len(self.key_list[l])):
        self.key_list[l][i] = self.key_list[l][i].to(device)
        self.key_list[l][i].requires_grad_(update_key)
    
    for l in range(len(self.prompt_list)):
      for i in range(len(self.prompt_list[l])):
        self.prompt_list[l][i] = self.prompt_list[l][i].to(device)
        self.prompt_list[l][i].requires_grad_(update_prompt)
  
  def addPrompt(self,add,device,embedding_layer=None):

    if add == 0:
      self.taskID_dict[len(self.taskID_dict.keys())] = self.total
      return
    self.total += add
    self.new = add

    if embedding_layer != None: # initialize key token with word embedding
      self.key_list = []
      for l in range(self.layer):
        ktmp = []

        for i in range(add):
          words = torch.randint(low=500,high=10000,size=(1,1),device=device)
          key = embedding_layer(words).squeeze().clone().detach()
          ktmp.append(key.requires_grad_(True))

        self.key_list[l] = self.key_list[l] + ktmp
    
    else:
      for l in range(len(self.key_list)):
        ktmp = [torch.randn(self.kdim,requires_grad=True,device=device) for i in range(add)]
        self.key_list[l] = self.key_list[l] + ktmp

    if embedding_layer != None: # initialize prompt token with word embedding
      for l in range(len(self.prompt_list)):
        ptmp = []

        for i in range(add):
          words = torch.randint(low=500,high=10000,size=(1,self.pnum),device=device)
          prompts = embedding_layer(words).squeeze().clone().detach()
          ptmp.append(prompts.requires_grad_(True))

        self.prompt_list[l] = self.prompt_list[l] + ptmp

    else:
      for l in range(len(self.prompt_list)):
        ptmp = [torch.randn((self.pnum,self.pdim),requires_grad=True,device=device) for i in range(add)]
        #ptmp = [torch.randn((self.pnum,self.pdim),requires_grad=True,device=device) for i in range(add-1)]+[torch.zeros((self.pnum,self.pdim),requires_grad=False,device=device)]
        self.prompt_list[l] = self.prompt_list[l] + ptmp
      
    self.taskID_dict[len(self.taskID_dict.keys())] = self.total
  
  def retainPool(self, path):
    tmp_pool = copy.deepcopy(self)
    for l in range(tmp_pool.layer):
      for num in range(tmp_pool.total):
        tmp_pool.key_list[l][num] = tmp_pool.key_list[l][num].detach().to('cpu')
        tmp_pool.prompt_list[l][num] = tmp_pool.prompt_list[l][num].detach().to('cpu')

    with open(path,'wb') as f:
      pickle.dump(tmp_pool,f)
    return
  
  def record_freq(self,selectedKey):
    selectedKey = selectedKey.reshape(-1).tolist()
    for k in selectedKey:
      self.key_freq_now[k] += 1
