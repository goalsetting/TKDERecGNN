# %% ########################
# 模块2：LLM特征增强器 modules/llm_enhancer.py
#############################
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from collections import OrderedDict, Counter
import re
import requests

# 配置API参数
DEEPSEEK_API_KEY = "YOUR_API_KEY"  # 替换为你的API密钥
API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

class PaneLLMEnhancer:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.text_encoder = SentenceTransformer("D:/Pytorch/EERecGNNs/all-MiniLM-L6-v2")
        self.cache = OrderedDict()
        self.pane_desc_cache = {}

        # 检测可用的文本特征
        self.text_features_available = self._detect_text_features()

        # 初始化物品元数据
        self.item_meta = self._load_item_metadata()
        # 初始化门控权重
        self.gate_weight = torch.nn.Parameter(torch.randn(config['embedding_size'],1))
        self.MLP = torch.nn.Parameter(torch.randn(384,config['embedding_size']))

    def generate_embeddings(self,text, model="deepseek-r1:1.5b"):
        url = "http://127.0.0.1:11434/api/embed"
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "input": text}
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def generate_completion(self,prompt, model="deepseek-r1:1.5b"):
        url = "http://127.0.0.1:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(url, headers=headers, json=data)
        return response.json().get('response', '')

    def clean_think_tags(self,text):
        """
        清理输出文本中的\\think标记
        """
        pattern = r'<\/?think.*?>'
        # cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text

    def _detect_text_features(self) -> bool:
        """检测数据集是否包含文本特征

        Returns:
            bool: 是否可用文本特征
        """
        if not hasattr(self.dataset, 'item_feat') or self.dataset.item_feat is None:
            return False

        item_feat_fields = set(self.dataset.field2source['item_id'])
        return any(f in item_feat_fields for f in ['movie_title', 'genre'])

    def _load_item_metadata(self):
        """加载物品元数据

        Returns:
            Dict[int, Dict]: 物品ID到元数据的映射
        """
        meta = {}
        for iid in range(1, self.dataset.item_num + 1):  # RecBole物品ID从1开始
            meta[iid] = {
                'interact_count': self._get_interact_count(iid).item(),
                'id': iid
            }

            if self.text_features_available:
                meta[iid].update({
                    'title': self.dataset.item_feat[iid].get('movie_title', f'Item_{iid}'),
                    'genre': self.dataset.item_feat[iid].get('genre', 'unknown')
                })

        return meta

    def _get_interact_count(self, iid: int) -> int:
        """获取物品交互次数

        Args:
            iid: 物品ID

        Returns:
            int: 交互次数
        """
        return (self.dataset.inter_feat['item_id'] == iid).sum()

    def _build_pane_prompt(self, item_ids):
        """构建窗格描述提示"""
        top_items = sorted(item_ids, key=lambda x: x[1], reverse=True)[:5]

        if self.text_features_available:
            item_descs = [
                f"{self.item_meta[iid]['title']} ({self.item_meta[iid]['genre']})"
                for iid, _ in top_items
            ]
            prompt = (
                    "根据以下用户群体的高频交互物品生成群体特征描述：\n"
                    "主要交互物品：\n- " + "\n- ".join(item_descs) + "\n"
                                                                    "请用不超过50字总结该群体的共同兴趣特征，需包含品类偏好和典型物品。"
            )
        else:
            # 无文本特征的提示模板
            item_ids = [f"ID_{iid}（交互次数：{count}）"
                        for iid, count in top_items]
            prompt = (
                    "根据用户群体的物品交互模式生成特征描述：\n"
                    "高频物品：\n- " + "\n- ".join(item_ids) + "\n"
                                                              "推断该群体的潜在兴趣特征（50字内）"
            )
        return prompt

    def query(self, prompt):
        desc_text = self.generate_completion(prompt=prompt,model="deepseek-r1:1.5b")
        desc_text = self.clean_think_tags(desc_text)
        return desc_text

    def get_pane_embedding(self, pane_items):
        """获取窗格文本嵌入"""
        # 生成物品频率分布
        item_counter = Counter(pane_items)
        total = sum(item_counter.values())
        item_freq = [(iid, count / total) for iid, count in item_counter.items()]

        # 检查缓存
        cache_key = hash(tuple(sorted(item_freq)))
        if cache_key in self.pane_desc_cache:
            return self.pane_desc_cache[cache_key]

        # LLM生成描述
        prompt = self._build_pane_prompt(item_freq)

        desc_text = self.generate_completion(prompt=prompt,model="deepseek-r1:1.5b")
        desc_text = self.clean_think_tags(desc_text)
        # response = ollama.generate(
        #     model='deepseek',
        #     prompt=prompt,
        #     options={'temperature': self.config['temperature'],
        #              'max_tokens': self.config['llm']['max_desc_length']}
        # )
        # desc_text = response['response']

        # 编码文本
        text_emb = self.text_encoder.encode(desc_text)
        text_emb = torch.FloatTensor(text_emb)

        # 更新缓存
        if len(self.cache) >= self.config['llm']['cache_size']:
            self.cache.popitem(last=False)
        self.pane_desc_cache[cache_key] = text_emb

        return text_emb

    def enhance_user_embeddings(self, user_emb, pane_embs, pane_labels):
        """增强用户嵌入"""
        # 收集每个用户的窗格嵌入
        pane_emb_dict = {}
        for label, emb in zip(np.unique(pane_labels), pane_embs):
            pane_emb_dict[label] = emb

        # 为每个用户分配对应窗格嵌入
        user_pane_embs = torch.stack([pane_emb_dict[label] for label in pane_labels])

        # 动态门控融合
        gate = torch.sigmoid(torch.matmul(user_emb, self.gate_weight))
        enhanced_emb = gate * user_emb + (1 - gate) * torch.mm(user_pane_embs,self.MLP)
        return enhanced_emb



    def generate_with_deepseek(self, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        data = {
            "model": "deepseek-chat",  # 根据实际模型名称调整
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,  # 控制响应长度
            "temperature": 0.7  # 控制创造性（0-1）
        }

        response = requests.post(API_ENDPOINT, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    # # 使用示例
    # if __name__ == "__main__":
    #     user_prompt = "帮我生成美甲产品的用户画像"  # 你的prompt
    #     result = generate_with_deepseek(user_prompt)
    #     print("API Response:\n", result)

