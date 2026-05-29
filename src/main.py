import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
from ultralytics import YOLO

def main():
    # 注意：假设你在项目根目录 (CV-Project2) 下运行 python src/main.py
    input_video_path = "project_video.mp4"
    output_video_path = "outputs/final_output.mp4" 
    yolo_model_path = "models/yolo11n.pt"

    # 1. 初始化 YOLO11 模型
    print("正在加载 YOLO11 模型...")
    # 第一次运行如果本地没有模型，它会自动下载 yolo11n.pt 到指定的路径
    yolo_model = YOLO(yolo_model_path)
    print("模型加载完成！")

    # 在 COCO 数据集中，对应的类别 ID 分别为: 2:car, 3:motorcycle, 5:bus, 7:truck
    vehicle_classes = [2, 3, 5, 7]

    # 打开视频流
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"开始处理视频: 分辨率 {width}x{height}, 帧率 {fps}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ==========================================
        # 车辆检测与跟踪模块 (YOLO11 + ByteTrack)
        # ==========================================
        # tracker="bytetrack.yaml" 指定使用 ByteTrack
        # persist=True 告诉模型在连续帧之间保持 track id
        # classes=vehicle_classes 过滤掉非车辆目标（比如路边的行人、交通标志）
        # verbose=False 关闭每帧打印，保持控制台整洁
        results = yolo_model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=vehicle_classes, 
            verbose=False
        )

        # 获取当前帧的预测结果
        result = results[0]
        
        # 确保画面中检测到了目标，并且 ByteTrack 分配了 ID
        if result.boxes is not None and result.boxes.id is not None:
            # 提取边界框坐标、跟踪ID、置信度和类别
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # 遍历每一个检测到的车辆并进行绘制
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = box
                cls_name = yolo_model.names[cls_id]

                # 绘制目标框 (绿色)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 拼接文本：类别 track_id 置信度
                label = f"{cls_name} ID:{track_id} Conf:{conf:.2f}"
                
                # 绘制文本背景（为了让文字在复杂背景下更清晰）
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20 - text_height), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # 绘制文本 (黑色字体)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 写入视频
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")

    cap.release()
    out.release()
    print("视频处理完成，已保存至:", output_video_path)

if __name__ == "__main__":
    main()