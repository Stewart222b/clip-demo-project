import spacy
from nltk.corpus import wordnet as wn
import numpy as np
from collections import defaultdict

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
        self.similarity_threshold = 0.65
        
        # 生成策略配置
        self.strategies = {
            'antonyms': True,
            'hypernyms': True,
            'hyponyms': False,
            'similarity': True,
            'domain_specific': []
        }
        
        # 领域特定负样本（可根据需求扩展）
        self.domain_negatives = {
            'clothing': ["naked", "undressed", "unclothed"],
            'vehicle': ["stationary", "parked", "motionless"]
        }

    def generate(self, text, top_n=5):
        """主生成方法"""
        if text in self.cache:
            return self.cache[text][:top_n]
            
        # 文本预处理
        doc = self.nlp(text.lower())
        negatives = []
        
        # 遍历每个有意义的token
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
                
            # 策略1：反义词挖掘
            if self.strategies['antonyms']:
                negatives += self._get_antonyms(token.text)
            
            # 策略2：上位词挖掘
            if self.strategies['hypernyms']:
                negatives += self._get_hypernyms(token.text)
            
            # 策略3：相似词检测
            if self.strategies['similarity']:
                negatives += self._get_similar_words(token)
        
        # 策略4：领域特定负样本
        negatives += self._get_domain_specific(text)
        
        # 后处理
        negatives = list(set(negatives))
        negatives = [f"a photo without {n}" for n in negatives if n]
        
        # 缓存结果
        self.cache[text] = negatives
        return negatives[:top_n]

    def _get_antonyms(self, word):
        """获取反义词"""
        antonyms = []
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        return list(set(antonyms))

    def _get_hypernyms(self, word):
        """获取上位词（更宽泛的概念）"""
        hypernyms = []
        for syn in wn.synsets(word):
            for hyper in syn.hypernyms():
                hypernyms.append(hyper.lemmas()[0].name())
        return list(set(hypernyms[:3]))  # 取前3个

    def _get_similar_words(self, token):
        """基于词向量找相似词"""
        similar = []
        for word in self.nlp.vocab:
            if word.has_vector and word.is_lower:
                similarity = token.similarity(word)
                if similarity > self.similarity_threshold:
                    similar.append(word.text)
        return list(set(similar))

    def _get_domain_specific(self, text):
        """领域特定负样本检测"""
        domain_neg = []
        doc = self.nlp(text)
        
        # 检测领域关键词
        for word in doc:
            if word.text in self.domain_negatives['clothing']:
                domain_neg += self.domain_negatives['clothing']
            elif word.text in self.domain_negatives['vehicle']:
                domain_neg += self.domain_negatives['vehicle']
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