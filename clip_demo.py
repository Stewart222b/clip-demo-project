import time
import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Lock, Event
from queue import Queue, LifoQueue
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from utils import PlaceholderEntry, Translator, cv2_puttext_chinese, letterbox
from negative_text_gen import NegativeTextGenerator

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 全局配置
MAX_FPS = 25       # 限制最大帧率
FRAME_SCALE = 0.5  # 画面缩放比例
DETECT_INTERVAL = 8  # 每3帧进行一次检测
PROBABILITY = 0.7
MAX_OBJECTS_TO_PROCESS = 40  # 限制每帧处理的最大目标数
BOX_DISPLAY_DURATION = 1.0  # 新增：检测框显示的最大时间（秒）
VIDEO_PATH = 0
# VIDEO_PATH = "/Users/Larry/Downloads/47516-451623701_medium.mp4"
# VIDEO_PATH = "/Users/Larry/Desktop/目标检测效果展示/视频演示/美国狩猎相机/source/test_video/Vehicle_01.mp4"
# VIDEO_PATH = "/Users/Larry/Downloads/2053100-hd_1920_1080_30fps.mp4"

# 共享资源
input_queue = Queue()
frame_queue = Queue(maxsize=3)    # 原始帧队列
yolo_queue = Queue(maxsize=1)     # YOLO处理队列
clip_queue = Queue(maxsize=40)    # CLIP处理队列
clip_tasks_queue = LifoQueue(maxsize=1)  # 新增：CLIP任务队列，只保存最新一帧的任务
result_queue = Queue(maxsize=1000)   # 结果队列
display_queue = Queue(maxsize=2)  # 用于显示的帧队列
exit_event = Event()
new_yolo_result = Event()  # 新增：标记有新的YOLO结果

# 添加锁来保护结果队列和检测对象
result_lock = Lock()
objects_lock = Lock()
clip_queue_lock = Lock()  # 新增：保护CLIP队列的锁

# 模型加载
yolo = YOLO("yolov8n.pt").to(device)
clip_model = CLIPModel.from_pretrained("/Users/Larry/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").to(device)
clip_processor = CLIPProcessor.from_pretrained("/Users/Larry/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
translate_model = Translator()
generator = NegativeTextGenerator()

# 性能统计
stats_lock = Lock()
perf_stats = {
    "yolo_fps": "N/A",
    "clip_fps": "N/A",
    "total_fps": "N/A",
    "objects_processed": 0,
    "skipped_frames": 0,  # 新增：记录跳过的帧数
    "frame_process_time": "N/A",  # 新增：每帧处理时间
    "last_frame_objects": 0       # 新增：上一帧处理的对象数量
}

# 添加帧任务协调结构
frame_tasks = {}
frame_tasks_lock = Lock()


class MainApplication:
    def __init__(self):
        # 主窗口
        self.root = tk.Tk()
        self.root.title("智能监控系统")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)

        self.entry_target_text = '' # 输入框文本
        self.current_target_text = '' # 当前搜索目标文本
        self.negative_texts = [] # 负文本列表

        self.entry_var = tk.StringVar() # 输入框文本变量
        self.target_var = tk.StringVar(value=self.current_target_text) # 当前搜索目标文本变量
        self.negative_var = tk.StringVar(value=self.negative_texts) # 负样本文本变量
        
        # 视频显示区域
        self.set_video_frame()
        
        # 控制面板
        self.set_control_frame()
        
        # 启动异步处理线程
        self.start_processing_threads()
        self.root.after(40, self.update_video)  # 25fps更新

        # 添加检测结果缓存，每个元素包含：((x1, y1, x2, y2), label, timestamp)
        self.detected_objects = []
        self.last_detection_time = time.time()


    def set_video_frame(self):
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(padx=10, pady=10, side=tk.BOTTOM)


    def set_control_frame(self):
        control_frame = tk.Frame(self.root, height=100)
        control_frame.pack(padx=10, pady=10, side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 输入组件
        entry = PlaceholderEntry(control_frame, textvariable=self.entry_var, width=20, placeholder="请输入搜索的目标 🔍", font=("Arial", 20))
        entry.place(x=10, y=10)

        # 按钮
        button = tk.Button(control_frame, text="搜索", command=self.update_target, font=("Arial", 25))
        button.place(x=100, y=60)
        
        # 性能监控
        self.stats_var = tk.StringVar()
        stats_label = tk.Label(control_frame, textvariable=self.stats_var, justify=tk.RIGHT, font=("Arial", 12))
        stats_label.place(relx=1.0, rely=0.0, anchor=tk.NE)

        # 当前搜索目标的标签
        current_target_label = tk.Label(control_frame, text="当前搜索目标:", font=("Arial", 20))
        current_target_label.place(relx=0.3, rely=0.1, anchor=tk.NW)

        # 显示当前搜索目标的值
        current_target_display = tk.Label(control_frame, textvariable=self.target_var, font=("Arial", 20))
        current_target_display.place(relx=0.3, rely=0.4, anchor=tk.NW)

        # 显示负样本标题
        negative_samples_label = tk.Label(control_frame, text="负样本:", font=("Arial", 20))
        negative_samples_label.place(relx=0.5, rely=0.1, anchor=tk.NW)

        # 显示负样本内容
        negative_samples_display = tk.Label(control_frame, textvariable=self.negative_var, 
                                           font=("Arial", 14), justify=tk.LEFT, wraplength=300)
        negative_samples_display.place(relx=0.6, rely=0.15, anchor=tk.NW)

        # 更新负样本方法
        def update_negative_samples():
            if self.negative_texts:
                # 只取前3个负样本显示
                neg_text = '\n'.join(self.negative_texts[:3])
                self.negative_var.set(neg_text)
            else:
                self.negative_var.set("无负样本")
                
        update_negative_samples()  # 初始化显示

    def start_processing_threads(self):
        """启动所有处理线程"""
        # 1. 视频采集线程
        Thread(target=self.frame_capture_worker, daemon=True).start()
        
        # 2. YOLO检测线程
        Thread(target=self.yolo_worker, daemon=True).start()
        
        # 3. CLIP调度器线程 - 新增
        Thread(target=self.clip_scheduler, daemon=True).start()
        
        # 4. CLIP处理线程 - 创建多个线程以提高并行度
        for _ in range(4):  # 可根据硬件调整线程数
            Thread(target=self.clip_worker, daemon=True).start()
            
        # 5. 结果处理与渲染线程
        Thread(target=self.render_worker, daemon=True).start()


    def update_target(self):
        if new_text := self.entry_var.get().strip():
            new_text_with_prompt = f"一张{new_text}的照片"
            en_text = translate_model.translate(new_text_with_prompt, 'zh2en')[0]
            self.negative_texts = generator.generate(en_text, top_n=5)
            print(f"英文文本：{en_text}")
            print(f"负文本：{self.negative_texts}")
            input_queue.put(en_text)
            self.negative_var.set("\n".join(self.negative_texts))
            self.target_var.set(self.entry_var.get())

    def frame_capture_worker(self):
        """线程1: 负责视频帧采集"""
        cap = cv2.VideoCapture(VIDEO_PATH)
        last_time = time.time()
        frame_count = 0

        try:
            while not exit_event.is_set():
                # 帧率控制
                elapsed = time.time() - last_time
                if elapsed < 1 / MAX_FPS: 
                    time.sleep(0.001)
                    continue
                
                # 计算总体FPS
                with stats_lock:
                    perf_stats["total_fps"] = f"{1 / elapsed:.1f}"
                
                last_time = time.time()
                
                # 读取帧
                ret, original_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # 处理输入更新
                if not input_queue.empty():
                    self.current_target_text = input_queue.get()
                
                # 缩放帧（先缩放用于显示）
                scaled_frame = cv2.resize(original_frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                
                # 将原始帧送入YOLO处理队列（每隔几帧）
                frame_count += 1
                if frame_count % DETECT_INTERVAL == 0:
                    if not yolo_queue.full():  # 防止处理过慢导致积压
                        yolo_queue.put((original_frame.copy(), scaled_frame.copy()))

                # 放入渲染队列
                if not frame_queue.full():
                    frame_queue.put(scaled_frame)
        finally:
            cap.release()
    
    def yolo_worker(self):
        """线程2: 负责YOLO目标检测"""
        while not exit_event.is_set():
            try:
                if yolo_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                original_frame, scaled_frame = yolo_queue.get()
                
                # YOLO检测
                yolo_start = time.time()
                results = yolo(original_frame, verbose=False, device=device)
                yolo_time = time.time() - yolo_start
                
                # 更新性能统计
                with stats_lock:
                    perf_stats["yolo_fps"] = f"{1 / yolo_time:.1f}"
                
                # 按置信度排序并限制处理目标数量
                boxes = []
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    boxes.append((box.xyxy[0], conf))
                
                # 按置信度降序排序
                boxes.sort(key=lambda x: x[1], reverse=True)
                
                # 创建一个帧ID来关联同一帧的所有目标
                frame_id = int(time.time() * 1000)  # 毫秒级时间戳作为帧ID
                
                # 获取当前帧要处理的目标数量
                targets_count = min(len(boxes), MAX_OBJECTS_TO_PROCESS)
                
                if targets_count > 0:
                    # 初始化帧任务计数器和结果列表
                    with frame_tasks_lock:
                        frame_tasks[frame_id] = {
                            'total': targets_count,  # 总任务数
                            'completed': 0,          # 已完成任务数
                            'results': [],           # 结果集合
                            'scaled_frame': scaled_frame.copy(),  # 保存当前帧用于后续处理
                            'start_time': None       # 新增：记录开始处理时间
                        }
                    
                    # 准备当前帧的所有目标处理任务
                    frame_tasks_list = []
                    for i, (box_tensor, _) in enumerate(boxes[:MAX_OBJECTS_TO_PROCESS]):
                        if i >= targets_count:
                            break
                        
                        x1, y1, x2, y2 = map(int, box_tensor)
                        
                        # 提取ROI并进行预处理
                        roi = original_frame[y1:y2, x1:x2]
                        if roi.size < 100:
                            # 跳过过小的目标
                            continue
                        
                        # 保存该目标的处理任务
                        frame_tasks_list.append((roi.copy(), (x1, y1, x2, y2), frame_id, i))
                    
                    # 把所有任务一次性提交到调度队列
                    if frame_tasks_list:
                        # 清空旧的CLIP任务队列
                        with clip_queue_lock:
                            while not clip_tasks_queue.empty():
                                try:
                                    clip_tasks_queue.get_nowait()
                                    with stats_lock:
                                        perf_stats["skipped_frames"] += 1
                                except:
                                    break
                            
                            # 提交新的任务列表
                            clip_tasks_queue.put((frame_id, frame_tasks_list))
                        
                        # 标记有新的YOLO结果等待处理
                        new_yolo_result.set()
                
            except Exception as e:
                print(f"YOLO处理错误: {e}")
                time.sleep(0.1)
    
    def clip_scheduler(self):
        """线程3: 负责调度CLIP任务，确保只处理最新的帧"""
        while not exit_event.is_set():
            try:
                # 等待新的YOLO结果
                new_yolo_result.wait(timeout=0.5)
                if exit_event.is_set():
                    break
                
                # 清除标志
                new_yolo_result.clear()
                
                # 检查是否有新任务
                if clip_tasks_queue.empty():
                    continue
                
                # 获取最新的一帧任务
                frame_id, tasks_list = clip_tasks_queue.get()
                
                # 记录开始处理这一帧的时间
                with frame_tasks_lock:
                    if frame_id in frame_tasks:
                        frame_tasks[frame_id]['start_time'] = time.time()
                        # 记录这一帧的目标总数
                        with stats_lock:
                            perf_stats["last_frame_objects"] = len(tasks_list)
                
                # 清空现有的CLIP队列
                with clip_queue_lock:
                    if not clip_queue.empty():
                        continue
                    
                    # 将新任务添加到CLIP处理队列
                    for task in tasks_list:
                        clip_queue.put(task)
                
            except Exception as e:
                print(f"CLIP调度器错误: {e}")
                time.sleep(0.1)
    
    def clip_worker(self):
        """线程4: 负责CLIP文本-图像匹配"""
        clip_times = []
        max_times = 10  # 用于计算平均CLIP处理时间
        
        while not exit_event.is_set():
            try:
                if clip_queue.empty():
                    time.sleep(0.01)
                    continue
                
                with clip_queue_lock:
                    if clip_queue.empty():
                        continue
                    roi, box_coords, frame_id, target_idx = clip_queue.get()
                
                # 检查该帧是否还在跟踪中
                with frame_tasks_lock:
                    if frame_id not in frame_tasks:
                        continue  # 帧已经被处理或清理
                
                x1, y1, x2, y2 = box_coords
                
                # 将坐标转换为缩放后的尺寸
                x1_scaled = int(x1 * FRAME_SCALE)
                y1_scaled = int(y1 * FRAME_SCALE)
                x2_scaled = int(x2 * FRAME_SCALE)
                y2_scaled = int(y2 * FRAME_SCALE)
                
                # CLIP处理
                clip_start = time.time()
                image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                input_texts = [self.current_target_text] + self.negative_texts
                if len(input_texts) <= 1 and input_texts[0] == "":
                    # 如果没有输入文本，则跳过处理
                    continue
                inputs = clip_processor(
                    text=input_texts,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                
                clip_time = time.time() - clip_start
                clip_times.append(clip_time)
                if len(clip_times) > max_times:
                    clip_times.pop(0)
                
                # 更新CLIP处理性能统计
                with stats_lock:
                    perf_stats["clip_fps"] = f"{1 / (sum(clip_times) / len(clip_times)):.1f}"
                    perf_stats["objects_processed"] += 1
                
                probs = outputs.logits_per_image.softmax(dim=1).squeeze()
                
                # 更新帧任务状态并检查是否是该帧的最后一个任务
                with frame_tasks_lock:
                    if frame_id in frame_tasks:
                        # 如果目标概率超过阈值，将结果添加到该帧的结果列表中
                        if probs[0].item() > PROBABILITY:
                            # 添加时间戳到结果中
                            current_time = time.time()
                            result_item = ((x1_scaled, y1_scaled, x2_scaled, y2_scaled), # 目标框坐标
                                          self.target_var.get().strip(), # 当前搜索目标文本
                                          probs[0].item(), #  # 置信度
                                          current_time)  # 添加时间戳
                            frame_tasks[frame_id]['results'].append(result_item)
                        
                        # 更新完成任务计数
                        frame_tasks[frame_id]['completed'] += 1
                        
                        # 检查该帧的所有目标是否都已处理完毕
                        if frame_tasks[frame_id]['completed'] >= frame_tasks[frame_id]['total']:
                            # 计算该帧处理的总时间
                            if 'start_time' in frame_tasks[frame_id]:
                                frame_process_time = time.time() - frame_tasks[frame_id]['start_time']
                                with stats_lock:
                                    perf_stats["frame_process_time"] = f"{frame_process_time:.3f}"
                                print(f"帧 {frame_id} 处理完成，共 {frame_tasks[frame_id]['total']} 个目标，耗时: {frame_process_time:.3f}秒")
                            
                            # 所有目标都已处理完，可以提交结果
                            with result_lock:
                                result_queue.put(frame_tasks[frame_id]['results'])
                            # 清理该帧的任务状态
                            del frame_tasks[frame_id]
                
            except Exception as e:
                print(f"CLIP处理错误: {e}")
                # 如果出错，也要更新计数，避免卡住
                try:
                    with frame_tasks_lock:
                        if frame_id in frame_tasks:
                            frame_tasks[frame_id]['completed'] += 1
                except:
                    pass
                time.sleep(0.1)
    
    def render_worker(self):
        """线程5: 负责渲染和结果合成"""
        while not exit_event.is_set():
            try:
                # 获取最新的渲染帧
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = frame_queue.get()
                current_time = time.time()
                
                # 使用锁保护结果队列的读取操作和检测对象的更新
                new_objects = []
                with result_lock:
                    if not result_queue.empty():
                        # 取最新的一帧结果
                        new_objects = result_queue.get()
                
                # 使用锁保护检测对象的更新
                with objects_lock:
                    # 如果有新检测结果，更新缓存和时间戳
                    if new_objects:
                        print(f"检测到目标: {len(new_objects)}")
                        self.detected_objects = new_objects
                        self.last_detection_time = current_time
                    
                    # 过滤掉超过显示时间的对象
                    updated_objects = []
                    for box, label, conf, timestamp in self.detected_objects:
                        if current_time - timestamp <= BOX_DISPLAY_DURATION:
                            updated_objects.append((box, label, conf, timestamp))
                    
                    # 更新过滤后的对象列表
                    self.detected_objects = updated_objects
                    
                    # 复制当前检测对象列表以避免在绘制过程中被修改
                    current_detected_objects = self.detected_objects.copy()
                
                # 绘制所有检测到的对象(新检测到的或缓存中的)
                for box, label, conf, timestamp in current_detected_objects:
                    # 计算剩余显示时间（秒）
                    remaining_time = BOX_DISPLAY_DURATION - (current_time - timestamp)
                    remaining_percent = remaining_time / BOX_DISPLAY_DURATION
                    
                    # 根据剩余时间修改颜色透明度
                    color_intensity = int(255 * min(1.0, remaining_percent * 2.0))  # 使颜色淡出更加明显
                    box_color = (0, 0, color_intensity)  # 红色，强度随时间减弱
                    
                    # 绘制检测框和标签
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    frame = trigger_alert(frame, f"{label} {conf:.2f}", box, color=(0, 0, color_intensity))
                
                # 准备性能统计
                with stats_lock:
                    stats_text = (
                        f"检测间隔: {DETECT_INTERVAL}帧\n"
                        f"YOLO FPS: {perf_stats['yolo_fps']}\n"
                        f"CLIP FPS: {perf_stats['clip_fps']}\n"
                        f"总体 FPS: {perf_stats['total_fps']}\n"
                        f"处理对象数: {perf_stats['objects_processed']}\n"
                        f"跳过的帧数: {perf_stats['skipped_frames']}\n"
                        f"当前显示对象: {len(current_detected_objects)}\n"
                        f"框显示时间: {BOX_DISPLAY_DURATION}秒\n"
                        f"每帧处理时间: {perf_stats['frame_process_time']}秒 ({perf_stats['last_frame_objects']}个目标)\n"
                        f"最后检测时间: {time.strftime('%H:%M:%S', time.localtime(self.last_detection_time))}"
                    )
                
                # 将结果放入显示队列
                if not display_queue.full():
                    # 预先转换为RGB，避免在UI线程中转换
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    display_queue.put((rgb_frame, stats_text))
                
            except Exception as e:
                print(f"渲染错误: {e}")
                time.sleep(0.1)

    def update_video(self):
        """更新视频显示"""
        try:
            if not display_queue.empty():
                frame, stats = display_queue.get()
                
                # 直接使用已转换为RGB的帧创建图像
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                
                self.video_frame.configure(image=img)
                self.video_frame.image = img  # 保持引用
                self.stats_var.set(stats)
        except Exception as e:
            print(f"视频更新错误: {e}")
            
        if not exit_event.is_set():
            self.root.after(40, self.update_video)  # 25fps更新
    
    def on_close(self):
        exit_event.set()
        self.root.destroy()

def trigger_alert(frame, text, box, color=(0, 0, 255), alpha=1):
    x1, y1, x2, y2 = box
    return cv2_puttext_chinese(
        img=frame,
        text=text,
        position=(x1, y1 - 35),
        font_size=30,
        color=color,
        color_alpha=alpha,
    )

if __name__ == "__main__":
    app = MainApplication()
    app.root.mainloop()