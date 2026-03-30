import torch
import torch.nn as nn
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len          # 每个token的长度
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"          # 多 GPU 模式，使用本地 rank
        else:
            self.device = f"cuda:{configs.gpu}"          # 单 GPU 模式，使用指定 GPU 
        print(self.device)
        
        self.qwen = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir, # configs.llm_ckp_dir,                            # 从指定目录加载预训练模型                             
            device_map=self.device,                         #  将模型加载到指定设备  
            dtype=torch.float16 if configs.use_amp else torch.float32,        # 根据 configs.use_amp（自动混合精度）选择数据类型（float16 或 float32）。
            local_files_only=True  # 强制使用本地文件，不从网络下载
        )
        self.hidden_dim_of_qwen = 4096
        self.mix = configs.mix_embeds           # 是否混合嵌入
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))          # 混合嵌入时的缩放因子       
        
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False                     # 冻结 Qwen 参数

        if configs.mlp_hidden_layers == 0:          # 使用简单的线性层 (nn.Linear) 作为编码器和解码器
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_qwen)                    # 线性编码器，将 token 转换为隐藏状态,token_len -> hidden_dim_of_qwen
            self.decoder = nn.Linear(self.hidden_dim_of_qwen, self.token_len)                      # 线性解码器，将隐藏状态转换为 token,hidden_dim_of_qwen -> token_len
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_qwen,                            # MLP 编码器
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,         # 使用 MLP 层，参数包括输入/输出维度、隐藏层维度 (mlp_hidden_dim)、层数 (mlp_hidden_layers)、Dropout 率 (dropout) 和激活函数 (mlp_activation)。
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_qwen, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 