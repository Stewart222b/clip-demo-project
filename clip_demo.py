import time
import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Lock, Event
from queue import Queue
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from utils import PlaceholderEntry, Translator, cv2_puttext_chinese, letterbox

# torch.set_num_threads(32)  # 根据CPU物理核心数调整

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 全局配置
entry_target_text = ""
current_target_text = "a person wearing a hat"
negative_texts = ["a picture of other", "a picture of Nothing", "a picture of man"]
MAX_FPS = 25       # 限制最大帧率
FRAME_SCALE = 0.5  # 画面缩放比例
DETECT_INTERVAL = 3  # 每3帧进行一次检测
PROBABILITY = 0.9

# 共享资源
input_queue = Queue()
frame_queue = Queue()  # 双缓冲队列
exit_event = Event()

# 模型加载
yolo = YOLO("yolov8n.pt").to(device)
clip_model = CLIPModel.from_pretrained("/Users/Larry/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").to(device)
clip_processor = CLIPProcessor.from_pretrained("/Users/Larry/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
translate_model = Translator()

class MainApplication:
    def __init__(self):
        # 主窗口
        self.root = tk.Tk()
        self.root.title("智能监控系统")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)

        self.entry_var = tk.StringVar(value=entry_target_text)
        self.target_var = tk.StringVar(value=current_target_text)
        
        # 视频显示区域
        self.set_video_frame()
        
        # 控制面板
        self.set_control_frame()
        
        # 启动线程
        Thread(target=self.camera_worker, daemon=True).start()
        self.root.after(100, self.update_video)  # 降低更新频率


    def set_video_frame(self):
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(padx=10, pady=10, side=tk.BOTTOM)


    def set_control_frame(self):
        control_frame = tk.Frame(self.root, height=100)
        control_frame.pack(padx=10, pady=10, side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 输入组件
        entry = PlaceholderEntry(control_frame, textvariable=self.entry_var, width=30, placeholder="请输入搜索的目标 🔍", font=("Arial", 20))
        entry.place(x=10, y=10)

        # 按钮
        button = tk.Button(control_frame, text="搜索", command=self.update_target, font=("Arial", 25))
        button.place(x=150, y=60)
        
        # 性能监控
        self.stats_var = tk.StringVar()
        stats_label = tk.Label(control_frame, textvariable=self.stats_var, justify=tk.LEFT, font=("Arial", 12))
        stats_label.place(relx=1.0, rely=0.0, anchor=tk.NE)

        # 当前搜索目标的标签
        current_target_label = tk.Label(control_frame, text="当前搜索目标:", font=("Arial", 20))
        current_target_label.place(relx=0.5, rely=0.2, anchor=tk.NW)

        # 显示当前搜索目标的值
        current_target_display = tk.Label(control_frame, textvariable=self.target_var, font=("Arial", 20))
        current_target_display.place(relx=0.5, rely=0.5, anchor=tk.NW)


    def update_target(self):
        if new_text := self.entry_var.get().strip():
            new_text_with_prompt = f"一张{new_text}的照片"
            en_text = translate_model.translate(new_text_with_prompt)[0]
            print(f"英文文本：{en_text}")
            input_queue.put(en_text)
            self.target_var.set(self.entry_var.get())

    def camera_worker(self):
        # cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("/Users/Larry/Desktop/目标检测效果展示/视频演示/美国狩猎相机/source/test_video/Vehicle_01.mp4")
        cap = cv2.VideoCapture("/Users/Larry/Downloads/2053100-hd_1920_1080_30fps.mp4")
        last_time = time.time()

        detect_counter = 0
        last_boxes = []
        last_labels = []
        yolo_fps = "N/A"  # 初始化变量
        clip_fps = "N/A"  # 初始化变量
        
        try:
            while not exit_event.is_set():
                # 帧率控制
                elapsed = time.time() - last_time
                if elapsed < 1 / MAX_FPS: 
                    time.sleep(0.001)
                    continue
                last_time = time.time()
                
                # 读取帧
                ret, original_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                 # 处理输入更新
                if not input_queue.empty():
                    global current_target_text
                    current_target_text = input_queue.get()
                
                # 缩放帧（先缩放用于显示）
                scaled_frame = cv2.resize(original_frame, (0,0), 
                                        fx=FRAME_SCALE, fy=FRAME_SCALE)
                
                detect_counter += 1
                process_detection = (detect_counter % DETECT_INTERVAL) == 0

                if process_detection:
                    # YOLO检测原始帧
                    yolo_start = time.time()
                    results = yolo(original_frame, verbose=False, device=device)
                    yolo_fps = 1 / (time.time() - yolo_start + 1e-5)
                    
                    current_boxes = []
                    current_labels = []
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 将坐标转换为缩放后的尺寸
                        x1_scaled = int(x1 * FRAME_SCALE)
                        y1_scaled = int(y1 * FRAME_SCALE)
                        x2_scaled = int(x2 * FRAME_SCALE)
                        y2_scaled = int(y2 * FRAME_SCALE)
                        
                        # CLIP处理
                        roi = original_frame[y1:y2, x1:x2]
                        roi = letterbox(roi, (224, 224))
                        
                        if roi.size < 100: continue
                        
                        clip_start = time.time()
                        image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        inputs = clip_processor(
                            text=[current_target_text, *negative_texts],
                            images=image,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                        
                        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
                        if probs[0].item() > PROBABILITY:
                            # 在缩放后的帧上绘制
                            cv2.rectangle(scaled_frame, 
                                        (x1_scaled, y1_scaled),
                                        (x2_scaled, y2_scaled),
                                        (0, 0, 255), 2)
                            scaled_frame = trigger_alert(scaled_frame, 
                                                        self.target_var.get().strip(),
                                                        (x1_scaled, y1_scaled, 
                                                        x2_scaled, y2_scaled))
                            current_boxes.append((x1_scaled, y1_scaled, 
                                                x2_scaled, y2_scaled))
                            current_labels.append(self.target_var.get().strip())
                        
                        clip_fps = 1 / (time.time() - clip_start + 1e-5)
                    
                    # 更新缓存
                    last_boxes = current_boxes
                    last_labels = current_labels
                else:
                    # 使用缓存的检测结果
                    for (x1, y1, x2, y2), label in zip(last_boxes, last_labels):
                        cv2.rectangle(scaled_frame, 
                                    (x1, y1), (x2, y2), 
                                    (0, 0, 255), 2)
                        scaled_frame = trigger_alert(scaled_frame, label, 
                                                    (x1, y1, x2, y2))
                
                # 放入队列
                if frame_queue.qsize() < 2:
                    frame_queue.put((
                        scaled_frame,
                        f"检测间隔: {DETECT_INTERVAL}帧\n" +
                        f"YOLO FPS: {yolo_fps if isinstance(yolo_fps, str) else f'{yolo_fps:.1f}'}\n" +
                        f"CLIP FPS: {clip_fps if isinstance(clip_fps, str) else f'{clip_fps:.1f}'}"
                    ))
        finally:
            cap.release()
    

    def update_video(self):
        # 更新画面和性能统计
        if not frame_queue.empty():
            frame, stats = frame_queue.get()
            # OpenCV转Tkinter格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            
            self.video_frame.configure(image=img)
            self.video_frame.image = img  # 保持引用
            self.stats_var.set(stats)
        
        if not exit_event.is_set():
            self.root.after(40, self.update_video)  # 25fps更新
    
    def on_close(self):
        exit_event.set()
        self.root.destroy()

def trigger_alert(frame, text, box):
    x1, y1, x2, y2 = box
    return cv2_puttext_chinese(
        img=frame,
        text=text,
        position=(x1, y1 - 35),
        font_size=30,
        color=(0, 0, 255)
    )

if __name__ == "__main__":
    app = MainApplication()
    app.root.mainloop()