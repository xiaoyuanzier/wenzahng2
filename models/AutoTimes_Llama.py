import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from layers.mlp import MLP

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len          # 每个token的长度
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"          # 多 GPU 模式，使用本地 rank
        else:
            self.device = f"cuda:{configs.gpu}"          # 单 GPU 模式，使用指定 GPU 
        print(self.device)

        # --- SSPN 新增参数与模块 ---
        self.num_groups = 4           # G: 分组数量 (确保 batch_size 能被其整除)
        self.n_fft = 64               # STFT 窗口大小
        self.hop_length = 16          # STFT 步长

        # 假设 seq_len=512, n_fft=64, hop=16, 得到的频谱图尺寸约为 33x33
        # 你需要根据实际输入长度运行一次并打印 spectrogram.shape 来微调此数值
        self.stft_flat_size = (self.n_fft // 2 + 1) * (configs.seq_len // self.hop_length + 1)
        self.causal_hidden_dim = 256  # 因果特征中间维度
        self.hidden_dim_of_llama = 4096 # Llama 隐藏维度
        
        # 1. 因果特征提取器 Phi_c
        self.phi = nn.Sequential(
            nn.Linear(self.stft_flat_size, self.causal_hidden_dim),
            nn.GELU(),
            nn.Linear(self.causal_hidden_dim, self.causal_hidden_dim)
        )

        # 2. 组内注意力打分器 (用于计算样本权重 alpha)
        self.attn_scorer = nn.Linear(self.causal_hidden_dim, 1, bias=False)

        # 3. 因果特征适配器 (用于将因果特征映射到 Llama 隐藏维度)
        self.causal_adapter = nn.Linear(self.causal_hidden_dim, self.hidden_dim_of_llama)
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir, 
            device_map=self.device,
            torch_dtype=torch.float16,  # 默认使用 float16 以节省显存
            low_cpu_mem_usage=True,     # 优化加载过程中的内存占用
            local_files_only=True
        )
        self.mix = configs.mix_embeds           # 是否混合嵌入
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))          # 混合嵌入时的缩放因子       
            # 时间标记投影器：将时间标记特征映射到 Llama 隐藏维度
            # 默认为 7 (ETTh 有 4 个, ETTm 有 5 个, 7 是最大常用值)
            self.mark_projector = nn.Linear(7, self.hidden_dim_of_llama)
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False                     # 冻结 LLaMA 参数

        if configs.mlp_hidden_layers == 0:          # 使用简单的线性层 (nn.Linear) 作为编码器和解码器
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)                      # 线性编码器，将 token 转换为隐藏状态,token_len -> hidden_dim_of_llama
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)                      # 线性解码器，将隐藏状态转换为 token,hidden_dim_of_llama -> token_len
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_llama,                            # MLP 编码器
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,         # 使用 MLP 层，参数包括输入/输出维度、隐藏层维度 (mlp_hidden_dim)、层数 (mlp_hidden_layers)、Dropout 率 (dropout) 和激活函数 (mlp_activation)。
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
    
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()                # 计算每变量的均值,并在计算梯度时 detach 掉
        x_enc = x_enc - means                                       # 对每个变量进行归一化，减去均值
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev                                              # 除以标准差        
        
        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]

        # ====================================================================
        # [SSPN 步骤 1]：1D 到 2D 的 STFT 时频变换
        # ====================================================================
        # 按照 AutoTimes 的通道独立性，展平 bs 和 n_vars 进行 STFT
        x_stft_input = x_enc.permute(0, 2, 1)
        # x_stft_input: [bs × nvars x seq_len]
        x_stft_input = x_stft_input.reshape(x_stft_input.shape[0] * x_stft_input.shape[1], -1)    
        stft_out = torch.stft(x_stft_input, n_fft=self.n_fft, hop_length=self.hop_length, 
                              return_complex=True, center=True)
        spectrogram = torch.log(torch.abs(stft_out) + 1e-8) # [bs*n_vars, F, T]

        # B. 分组因果特征重构 (Group-wise Recomposition)
        F_dim, T_dim = spectrogram.shape[1], spectrogram.shape[2]
        # 拆回 [bs, n_vars, F*T]
        S_flat = spectrogram.reshape(bs, n_vars, -1)
        k = bs // self.num_groups # 每组样本数
        if k > 0:
            valid_bs = k * self.num_groups
            # 变形为分组格式: [G, k, n_vars, 特征]
            grouped_S = S_flat[:valid_bs].reshape(self.num_groups, k, n_vars, -1)
            
            # 提取特征 Phi(S)
            feat = self.phi(grouped_S) # [G, k, n_vars, causal_dim]
            
            # 计算组内注意力 alpha 并聚合生成“因果核心”
            attn_weights = torch.softmax(self.attn_scorer(feat), dim=1) # 对每组内的 k 个样本做 softmax
            causal_core = torch.sum(attn_weights * feat, dim=1) # [G, n_vars, causal_dim]
            
            # 广播回原始 Batch 尺寸
            causal_features = causal_core.unsqueeze(1).expand(-1, k, -1, -1).reshape(valid_bs * n_vars, -1)
            
            # [新增处理]：处理余下的样本 (当 bs 不能被 num_groups 整除时)
            if valid_bs < bs:
                remainder_S = S_flat[valid_bs:] # [bs-valid_bs, n_vars, F*T]
                remainder_feat = self.phi(remainder_S.reshape(-1, remainder_S.shape[-1])) # [ (bs-valid_bs)*n_vars, causal_dim ]
                causal_features = torch.cat([causal_features, remainder_feat], dim=0)
                valid_bs = bs # 关键：恢复为原始 bs
        else:
            # Batch 太小时的兼容处理
            valid_bs = bs
            causal_features = self.phi(S_flat.reshape(bs * n_vars, -1))

        # C. 跨模态投影生成 Causal Prompt
        # [valid_bs * n_vars, 1, llama_hidden_dim]
        causal_prompt = self.causal_adapter(causal_features).unsqueeze(1)

        # ---------- 3. [AutoTimes 原始逻辑] 1D Patching 与 Embedding ----------
        # 准备 1D 输入: [valid_bs * n_vars, seq_len]
        x_enc_1d = x_enc[:valid_bs].permute(0, 2, 1).reshape(valid_bs * n_vars, -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc_1d.unfold(dimension=-1, size=self.token_len, step=self.token_len)       # 分段，将每个变量的序列长度 seq_len 分成多个 token_len 长度的片段
        token_num = fold_out.shape[1]                       # token 数量
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llama]
        times_embeds = self.encoder(fold_out)

        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            # 1. 获取标记并对齐 Batch
            x_mark_valid = x_mark_enc[:valid_bs] # [bs, seq_len, d_mark]
            
            # 2. 将时间维度从 seq_len 对齐到 token_num
            # 使用自适应平均池化，无论 seq_len 是多少，都能准确得到 token_num 个点
            # 输入: [bs, seq_len, d_mark] -> 变换维度: [bs, d_mark, seq_len]
            x_mark_tmp = x_mark_valid.permute(0, 2, 1)
            x_mark_pool = F.adaptive_avg_pool1d(x_mark_tmp, token_num).permute(0, 2, 1) # [bs, token_num, d_mark]
            
            # 3. 投影到 Llama 的隐藏维度
            # 确保输入维度匹配 mark_projector (7 维)
            d_mark = x_mark_pool.shape[-1]
            if d_mark < 7:
                # 补齐到 7 维
                padding = torch.zeros([valid_bs, token_num, 7 - d_mark]).to(x_mark_pool.device)
                x_mark_pool = torch.cat([x_mark_pool, padding], dim=-1)
            elif d_mark > 7:
                # 截断到 7 维
                x_mark_pool = x_mark_pool[:, :, :7]
            
            mark_embeds = self.mark_projector(x_mark_pool) # [bs, token_num, hidden_dim]
            
            # 4. 广播到所有变量通道并相加
            # [bs, token_num, hidden_dim] -> [bs, 1, token_num, hidden_dim] -> [bs, n_vars, token_num, hidden_dim]
            mark_embeds = mark_embeds.unsqueeze(1).repeat(1, n_vars, 1, 1).reshape(valid_bs * n_vars, token_num, -1)
            
            mark_embeds = mark_embeds / mark_embeds.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * mark_embeds

        # ---------- 4. [SSPN 注入] 拼接 Prompt 并调用 LLaMA ----------
        # 将 2D 因果提示词拼在 1D 时域 Token 之前
        # final_embeds: [bs * nvars, 1 + token_num, hidden_dim]
        final_embeds = torch.cat([causal_prompt, times_embeds], dim=1)

        # outputs: [bs * n_vars x (1 + token_num) x hidden_dim_of_llama]
        outputs = self.llama.model(                 # 直接传入嵌入，跳过 tokenization 过程
            inputs_embeds=final_embeds)[0]
        # 裁剪掉最前面的 Causal Prompt 对应的输出，只保留 1D Token 对应的输出
        outputs = outputs[:, 1:, :]
        
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(valid_bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        # 将预测结果乘以标准差并加上均值，恢复原始尺度。
        dec_out = dec_out * \
            (stdev[:valid_bs, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))          
        dec_out = dec_out + \
            (means[:valid_bs, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)