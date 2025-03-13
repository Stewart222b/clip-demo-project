import spacy
from nltk.corpus import wordnet as wn
import numpy as np
from collections import defaultdict
import re

class NegativeTextGenerator:
    """
    NegativeTextGenerator 类用于生成基于给定文本的负样本描述。
    
    该类通过多种策略（如反义词、上位词、相似词和领域特定负样本）生成与输入文本相关的负样本描述。
    生成的负样本可以用于图像描述、文本生成等任务，以提供多样化的文本输出。
    
    核心功能：
    - 生成负样本描述（generate 方法）
    - 获取反义词（_get_antonyms 方法）
    - 获取上位词（_get_hypernyms 方法）
    - 获取相似词（_get_similar_words 方法）
    - 获取领域特定负样本（_get_domain_specific 方法）
    
    使用示例：
    
    构造函数参数：
    无
    
    特殊使用限制或潜在的副作用：
    - 该类依赖于 spaCy 和 NLTK 中的 WordNet，需要在环境中安装这些库。
    - 生成的负样本描述可能不总是准确或相关，特别是对于罕见或特定领域的词汇。
    - 缓存系统可能会占用较多内存，特别是在处理大量不同文本时。
    """
    
    def __init__(self):
        # 加载语言模型
        self.nlp = spacy.load("en_core_web_md")
        
        # 初始化缓存系统
        self.cache = defaultdict(list)
        self.similarity_threshold = 0.85  # 增加相似度阈值，减少无关词汇
        
        # 生成策略配置
        self.strategies = {
            'antonyms': True,
            'hypernyms': True,
            'hyponyms': True,  # 启用下位词
            'similarity': True,
            'domain_specific': True
        }
        
        # 扩展领域特定负样本
        self.domain_negatives = {
            'clothing': {
                'keywords': ["hat", "shirt", "dress", "jacket", "coat", "pants", "shoes", "clothes", "wearing", "outfit", "fashion"],
                'negatives': ["naked", "undressed", "unclothed", "without clothes", "bareheaded", "no hat", "hatless"]
            },
            'vehicle': {
                'keywords': ["car", "vehicle", "truck", "bus", "motorcycle", "bike", "driving", "road", "highway"],
                'negatives': ["stationary", "parked", "motionless", "no vehicle", "empty road", "without cars"]
            },
            'person': {
                'keywords': ["person", "people", "man", "woman", "child", "boy", "girl", "human"],
                'negatives': ["empty", "uninhabited", "deserted", "no people", "without humans", "unpopulated"]
            },
            'accessory': {
                'keywords': ["handbag", "bag", "purse", "backpack", "wallet", "holding", "carrying"],
                'negatives': ["empty-handed", "without accessories", "no bags", "nothing held"]
            }
        }
        
        # 预先计算常用词的反义词和上位词，提高性能
        self.common_antonyms = {
            'happy': ['sad', 'unhappy'],
            'big': ['small', 'tiny', 'little'],
            'tall': ['short', 'small'],
            'wearing': ['not wearing', 'without'],
            'holding': ['empty-handed', 'not holding'],
            'open': ['closed', 'shut'],
            'new': ['old', 'used', 'worn'],
            'full': ['empty', 'vacant'],
            'bright': ['dark', 'dim'],
            'clean': ['dirty', 'messy'],
            'wet': ['dry'],
            'hot': ['cold', 'cool'],
            'many': ['few', 'none'],
            'inside': ['outside', 'outdoors'],
            'young': ['old', 'elderly', 'aged']
        }
        
        # 无效或不相关词过滤表
        self.filter_words = ['have', 'be', 'do', 't.', 'c.', 'function', 'refresh', 'sum', 'means', 'direct']

    def generate(self, text, top_n=5):
        """
        为输入文本生成负样本描述。
        
        Args:
            text (str): 输入文本，用于生成负样本。
            top_n (int, optional): 返回的负样本数量。默认为5。
            
        Returns:
            list: 负样本描述列表。
        """
        if text in self.cache:
            return self.cache[text][:top_n]
            
        # 文本预处理
        doc = self.nlp(text.lower())
        negatives = []
        
        # 提取关键词和词性
        key_nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        key_verbs = [token.text for token in doc if token.pos_ == 'VERB']
        key_adjs = [token.text for token in doc if token.pos_ == 'ADJ']
        
        processed_words = set()  # 避免对相同的词多次处理
        
        # 首先处理关键名词
        for noun in key_nouns:
            if noun in processed_words:
                continue
                
            processed_words.add(noun)
            
            # 策略1：反义词挖掘
            if self.strategies['antonyms']:
                antonyms = self._get_antonyms(noun)
                negatives.extend([f"without {noun}", f"no {noun}"])
                negatives.extend([f"{ant}" for ant in antonyms if ant not in self.filter_words])
            
            # 策略2：上位词和下位词挖掘
            if self.strategies['hypernyms']:
                hypernyms = self._get_hypernyms(noun)
                negatives.extend([f"without {h}" for h in hypernyms if h not in self.filter_words])
        
        # 处理关键动词
        for verb in key_verbs:
            if verb in processed_words:
                continue
                
            processed_words.add(verb)
            
            # 动词反义词
            ant_verbs = self._get_antonyms(verb)
            negatives.extend([f"not {verb}ing", f"without {verb}ing"])
            negatives.extend([f"{ant}" for ant in ant_verbs if ant not in self.filter_words])
            
        # 处理形容词
        for adj in key_adjs:
            if adj in processed_words:
                continue
                
            processed_words.add(adj)
            
            # 形容词反义词
            ant_adjs = self._get_antonyms(adj)
            negatives.extend([f"not {adj}", f"non-{adj}"])
            negatives.extend([ant for ant in ant_adjs if ant not in self.filter_words])
        
        # 策略4：领域特定负样本
        if self.strategies['domain_specific']:
            domain_negs = self._get_domain_specific(text)
            negatives.extend(domain_negs)
        
        # 后处理
        negatives = [n for n in negatives if n and len(n) > 2 and n not in self.filter_words]
        negatives = list(set(negatives))  # 去重
        
        # 构造完整的否定描述
        formatted_negatives = []
        for neg in negatives:
            if neg.startswith("without") or neg.startswith("no ") or neg.startswith("not "):
                formatted_negatives.append(f"a photo {neg}")
            else:
                formatted_negatives.append(f"a photo without {neg}")
        
        # 缓存结果
        self.cache[text] = formatted_negatives
        return formatted_negatives[:top_n]

    def _get_antonyms(self, word):
        """获取反义词"""
        # 首先检查常用词表
        if word in self.common_antonyms:
            return self.common_antonyms[word]
            
        antonyms = []
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonym = lemma.antonyms()[0].name().replace('_', ' ')
                    if antonym not in self.filter_words:
                        antonyms.append(antonym)
        
        # 如果没有找到反义词，尝试添加否定前缀
        if not antonyms and len(word) > 3:
            if not word.startswith('un'):
                antonyms.append(f"un{word}")
            if not word.startswith('non'):
                antonyms.append(f"non-{word}")
                
        return list(set(antonyms))

    def _get_hypernyms(self, word):
        """获取上位词（更宽泛的概念）"""
        hypernyms = []
        for syn in wn.synsets(word):
            for hyper in syn.hypernyms():
                name = hyper.lemmas()[0].name().replace('_', ' ')
                if name not in self.filter_words and len(name) > 2:
                    hypernyms.append(name)
        return list(set(hypernyms[:2]))  # 减少数量，提高质量

    def _get_hyponyms(self, word):
        """获取下位词（更具体的概念）"""
        hyponyms = []
        for syn in wn.synsets(word):
            for hypo in syn.hyponyms():
                name = hypo.lemmas()[0].name().replace('_', ' ')
                if name not in self.filter_words and len(name) > 2:
                    hyponyms.append(name)
        return list(set(hyponyms[:2]))

    def _get_similar_words(self, token):
        """基于词向量找相似词（已优化，减少不相关词）"""
        similar = []
        
        # 只对重要词性进行相似度计算
        if token.pos_ not in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
            return []
            
        # 使用更有选择性的方法，避免遍历整个词汇表
        most_similar = sorted(
            [(w, token.similarity(w)) for w in self.nlp.vocab 
             if w.has_vector and w.is_lower and w.text != token.text 
             and w.text not in self.filter_words and len(w.text) > 2],
            key=lambda item: -item[1]
        )
        
        # 只选取最相关的前几个词
        for word, sim in most_similar[:3]:
            if sim > self.similarity_threshold:
                similar.append(word.text)
        
        return similar

    def _get_domain_specific(self, text):
        """领域特定负样本检测（改进版）"""
        domain_neg = []
        doc = self.nlp(text.lower())
        text_lower = text.lower()
        
        # 检测领域匹配
        for domain, data in self.domain_negatives.items():
            # 检查关键词是否存在于文本中
            if any(keyword in text_lower for keyword in data['keywords']):
                domain_neg.extend(data['negatives'])
        
        return domain_neg

# 使用示例
if __name__ == "__main__":
    generator = NegativeTextGenerator()
    
    test_cases = [
        "a person wearing a hat",
        "red car on the road",
        "woman holding a handbag"
    ]
    
    for text in test_cases:
        negatives = generator.generate(text)
        print(f"Input: {text}")
        print(f"Generated negatives: {negatives}\n")