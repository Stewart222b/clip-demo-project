import cv2
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk

class PlaceholderEntry(ttk.Entry):
    """
    PlaceholderEntry 类是一个带有占位符功能的文本输入框，继承自 ttk.Entry。

    功能:
    - 在文本框为空时显示占位符文本。
    - 当文本框获得焦点时，自动清除占位符文本。
    - 当文本框失去焦点且为空时，自动恢复占位符文本。

    使用示例:

    构造函数参数:
    - master: 父组件，通常是 Tkinter 的窗口或框架。
    - placeholder: 占位符文本，默认为空字符串。
    - **kwargs: 其他传递给 ttk.Entry 的参数。

    注意:
    - 占位符文本的颜色默认为灰色，可以通过修改代码来改变。
    - 该类适用于需要提示用户输入的文本框场景。
    """
    
    def __init__(self, master=None, placeholder="", **kwargs):
        """
        初始化 PlaceholderEntry 类的实例。

        参数:
        - master: 父组件。
        - placeholder: 占位符文本，默认为空字符串。
        - **kwargs: 其他传递给 ttk.Entry 的参数。
        """
        super().__init__(master, **kwargs)  # 调用父类的初始化方法
        self.placeholder = placeholder  # 设置占位符文本
        self.default_fg_color = self.cget("foreground")  # 保存默认的前景色
        self.bind("<FocusIn>", self.on_focus_in)  # 绑定焦点进入事件
        self.bind("<FocusOut>", self.on_focus_out)  # 绑定焦点离开事件
        self.put_placeholder()  # 调用 put_placeholder 方法显示占位符

    def put_placeholder(self):
        """
        在文本框中插入占位符文本，并将前景色设置为灰色。
        """
        self.insert(0, self.placeholder)
        self.config(foreground="gray")

    def on_focus_in(self, event):
        """
        当文本框获得焦点时触发的事件处理函数。

        参数:
        - event: 焦点事件对象。
        """
        if self.get() == self.placeholder:
            self.delete(0, tk.END)
            self.config(foreground=self.default_fg_color)

    def on_focus_out(self, event):
        """
        当文本框失去焦点时触发的事件处理函数。

        参数:
        - event: 焦点事件对象。
        """
        if not self.get():
            self.put_placeholder()


class Translator:
    """
    Translator 类用于中英文互译。

    该类提供了两种翻译方向：英语到中文（EN2ZH）和中文到英语（ZH2EN）。
    它通过加载预训练的MarianMT模型和对应的tokenizer来实现翻译功能。

    核心功能：
    - 翻译文本：translate方法可以根据指定的翻译方向（EN2ZH或ZH2EN）将输入文本进行翻译。

    使用示例：

    构造函数参数：
    - 无需参数，构造函数内部会自动加载预训练模型和tokenizer。

    特殊使用限制或潜在的副作用：
    - 该类依赖于本地或预训练模型的路径，确保模型和tokenizer已正确下载并放置在指定路径。
    - 翻译结果可能受限于预训练模型的准确性和覆盖范围。
    """
    EN2ZH = 'en2zh'
    ZH2EN = 'zh2en'

    def __init__(self):
        """
        构造函数，用于初始化Translator对象。

        - 加载英语到中文和中文到英语的预训练模型及其对应的tokenizer。
        """
        # 模型名称，如果模型已经下载到本地，可以直接指定路径
        en2zh_model_name = "/Users/Larry/Projects/CLIP/translate/opus-mt-en-zh"
        zh2en_model_name = "/Users/Larry/Projects/CLIP/translate/opus-mt-zh-en"
        # 加载tokenizer ，用于将用户输入的文本转化为数字形式，将模型输出的数字形式转化为中文形式
        self.en2zh_tokenizer = MarianTokenizer.from_pretrained(en2zh_model_name)
        self.zh2en_tokenizer = MarianTokenizer.from_pretrained(zh2en_model_name)
        # 加载模型
        self.en2zh_model = MarianMTModel.from_pretrained(en2zh_model_name)
        self.zh2en_model = MarianMTModel.from_pretrained(zh2en_model_name)

    def translate(self, src_text: str, format: str):
        """
        根据指定的翻译方向将输入文本进行翻译。

        参数：
        - src_text (str): 需要翻译的源文本。
        - format (str): 翻译方向，可以是 EN2ZH（英语到中文）或 ZH2EN（中文到英语）。

        返回：
        - list[str]: 翻译后的文本列表。
        """
        if format == 'en2zh':
            # 开始翻译：使用英语到中文的模型和分词器进行翻译
            translated = self.en2zh_model.generate(**self.en2zh_tokenizer(src_text, return_tensors="pt", padding=True))
            # 返回结果：将翻译后的token解码为文本，跳过特殊字符
            r = [self.en2zh_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        elif format == 'zh2en':
            # 开始翻译：使用中文到英语的模型和分词器进行翻译
            translated = self.zh2en_model.generate(**self.zh2en_tokenizer(src_text, return_tensors="pt", padding=True))
            # 返回结果：将翻译后的token解码为文本，跳过特殊字符
            r = [self.zh2en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return r


def letterbox(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    将图像缩放并填充到目标尺寸，保持图像的长宽比。

    参数:
    image (numpy.ndarray): 输入的图像数组。
    target_size (tuple): 目标尺寸，格式为 (宽度, 高度)，默认为 (640, 640)。
    color (tuple): 填充颜色，默认为 (114, 114, 114)。

    返回:
    numpy.ndarray: 处理后的图像数组，尺寸为 target_size。

    步骤:
    1. 获取原始图像的尺寸。
    2. 计算缩放后的新尺寸，保持长宽比。
    3. 缩放图像到新尺寸。
    4. 计算填充边界。
    5. 添加填充以使图像达到目标尺寸。
    """
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    new_w = int(w * min(target_size[0]/w, target_size[1]/h))
    new_h = int(h * min(target_size[0]/w, target_size[1]/h))
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算填充边界
    dw = (target_size[0] - new_w) // 2
    dh = (target_size[1] - new_h) // 2
    top, bottom = dh, dh + (target_size[1] - new_h) % 2
    left, right = dw, dw + (target_size[0] - new_w) % 2
    
    # 添加填充
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return padded #  返回填充后的图像


def cv2_puttext_chinese(img, text, position, font_size, color=(255, 255, 255), color_alpha=1, font_path="/Users/Larry/Downloads/pingfangziti_downcc/pingfangziti/苹方黑体-中粗-简.ttf",  bg_color=(50, 50, 50), bg_alpha=0.6):
    """
    Draw Chinese text on an OpenCV image with an optional background box.

    Parameters:
        img (numpy.ndarray): Input OpenCV image.
        text (str): Text to be drawn.
        position (tuple): Position of the text in the format (x, y).
        font_size (int): Font size.
        color (tuple, optional): Text color, default is white (255, 255, 255).
        color_alpha (float, optional): Text transparency, default is 1 (opaque).
        font_path (str, optional): Path to the font file, default is 苹方黑体-中粗-简.ttf.
        bg_color (tuple, optional): Background color, default is gray (50, 50, 50). If None, no background box is drawn.
        bg_alpha (float, optional): Background transparency, default is 0.6.
    Returns:
        numpy.ndarray: OpenCV image with the drawn text.
    """
    # 转换颜色通道顺序
    color_rgb = color[::-1]  # BGR -> RGB
    bg_color_rgb = bg_color[::-1] if bg_color else None
    
    # 将OpenCV图像转换为PIL格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 计算文字尺寸
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # 添加背景框
    if bg_color:
        # 计算背景框位置
        x, y = position
        margin = 3
        bg_box = (
            x - margin,
            y - margin,
            x + text_w + margin*2,
            y + text_h + margin*2
        )
        # 绘制半透明背景
        draw.rectangle(bg_box, fill=(bg_color_rgb + (int(255 * bg_alpha),)))
    
    # 绘制文字
    draw.text(position, text, font=font, fill=(color_rgb + (int(255 * color_alpha),)))
    
    # 转换回OpenCV格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img
