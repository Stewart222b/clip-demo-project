# 智能监控系统 (CLIP-based Intelligent Monitoring)

基于CLIP和YOLO的实时视觉搜索与监控系统，允许用户通过自然语言描述搜索视频中的目标对象。

![系统界面预览](/assets/preview.png)

## 项目简介

该项目结合了YOLO物体检测与OpenAI的CLIP(Contrastive Language-Image Pretraining)模型，实现了一个创新的视频监控系统，可以：

- 实时检测视频中的多个目标物体
- 通过自然语言描述搜索特定目标
- 支持中英文双语输入和翻译
- 自动生成负样本文本以提高检测精度
- 提供高效的多线程处理框架

## 功能特点

- **实时物体检测**: 利用YOLO v8模型进行高效物体检测
- **自然语言搜索**: 使用CLIP模型实现文本到图像的匹配搜索
- **双语支持**: 内置中英文翻译，支持中文输入
- **负样本生成**: 自动生成相关负样本以提高匹配精度
- **高性能架构**: 多线程设计，支持实时处理与显示
- **友好界面**: 简洁直观的图形界面，显示关键性能指标
- **实时性能监控**: 提供FPS、处理时间等性能指标实时显示

## 系统架构

该系统采用多线程架构，主要包含以下组件：

1. **视频采集线程**: 负责从摄像头或视频文件读取帧
2. **YOLO检测线程**: 执行物体检测并提取感兴趣区域(ROI)
3. **CLIP调度器线程**: 管理CLIP处理任务，确保只处理最新帧
4. **CLIP处理线程**: 执行文本-图像匹配操作(多线程)
5. **渲染线程**: 合成并显示结果帧

## 安装要求

### 系统要求
- Python 3.8+
- CUDA兼容的GPU (推荐用于实时处理)
- 或苹果M系列芯片(支持MPS加速)

### 依赖库
```bash
pip install -r requirements.txt
```

主要依赖:
- torch
- transformers
- ultralytics (YOLO)
- opencv-python
- Pillow
- spacy
- nltk
- tkinter

### 模型下载
- YOLO v8模型 (`yolov8n.pt`)
- CLIP模型 (将自动下载或指定本地路径)
- Opus-MT翻译模型 (需放置在`translate`目录)

## 使用方法

1. **启动程序**:
```bash
python clip_demo.py
```

2. **选择视频源**:
修改`clip_demo.py`中的`VIDEO_PATH`变量:
- `0`: 使用默认摄像头
- 文件路径: 使用特定视频文件

3. **搜索目标**:
- 在输入框中输入您想搜索的目标描述(如"戴帽子的人")
- 点击"搜索"按钮或按回车键
- 系统会自动翻译描述并生成负样本
- 检测结果将在视频上实时显示

## 配置参数

可在`clip_demo.py`文件顶部修改以下全局配置:

- `MAX_FPS`: 限制最大帧率
- `FRAME_SCALE`: 画面缩放比例
- `DETECT_INTERVAL`: 检测间隔(帧数)
- `PROBABILITY`: 匹配置信度阈值
- `MAX_OBJECTS_TO_PROCESS`: 每帧最大处理目标数
- `BOX_DISPLAY_DURATION`: 检测框显示的最大时间(秒)
- `VIDEO_PATH`: 视频源

## 项目结构

- `clip_demo.py`: 主程序，包含UI和多线程处理逻辑
- `utils.py`: 工具函数，包含翻译器、文本渲染等功能
- `negative_text_gen.py`: 负样本文本生成器
- `translate/`: 包含中英互译的预训练模型

## 性能调优

- 调整`DETECT_INTERVAL`可平衡检测频率与系统负载
- 修改`MAX_OBJECTS_TO_PROCESS`限制每帧处理的目标数量
- 在`start_processing_threads()`中可调整CLIP处理线程数

## 许可证

[项目许可证信息]

## 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 作者

Larry
