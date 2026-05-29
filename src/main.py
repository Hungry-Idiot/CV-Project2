import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    将图像按原比例缩放，并用纯色填充到指定尺寸 (YOLOP 预处理必须)
    """
    shape = img.shape[:2]  # 当前形状 [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高方向的 padding
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:  # 缩放图像
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def main():
    input_video_path = "project_video.mp4"
    output_video_path = "outputs/final_output.mp4" 
    yolo_model_path = "models/yolo11n.pt"
    yolop_model_path = "models/yolop-640-640.onnx"

    # ================= 1. 初始化模型 =================
    print("正在加载 YOLO11 与 YOLOP 模型...")
    yolo_model = YOLO(yolo_model_path)
    # 使用 ONNXRuntime 加载 YOLOP，优先尝试 GPU 加速
    ort_session = ort.InferenceSession(yolop_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("模型加载完成！")

    vehicle_classes = [2, 3, 5, 7] # 车辆相关的 COCO 类别

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

    # ================= 2. 提前计算 p63 裁剪掩码 =================
    # 报告要求的截取范围：
    # - 高度：63%以下 (即 Y 坐标从 height*0.63 到 height)
    # - 顶部宽度：40% ~ 60%
    # - 底部宽度：16% ~ 90%
    roi_pts = np.array([[
        (int(width * 0.16), height), 
        (int(width * 0.40), int(height * 0.63)), 
        (int(width * 0.60), int(height * 0.63)), 
        (int(width * 0.90), height)
    ]], dtype=np.int32)
    
    # 创建一个与原图同尺寸的黑白多边形遮罩
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_pts, 255)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ==========================================
        # 模块 A：YOLOP 车道线分割
        # ==========================================
        # 1. 预处理
        img_resized, r, (dw, dh) = letterbox(frame, new_shape=(640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # OpenCV 默认 BGR，需转 RGB
        img_norm = img_rgb.astype(np.float32) / 255.0          # 像素归一化到 [0,1]
        
        # ImageNet 标准化
        img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 强制转回 float32
        img_norm = img_norm.astype(np.float32)

        # 转换为 NCHW 格式 [1, 3, 640, 640]
        img_tensor = img_norm.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # 2. 模型推理
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
        outputs = ort_session.run(None, ort_inputs)
        
        # YOLOP 输出包含多个头，通常索引为 2 的是车道线 (lane_line_seg)
        lane_seg_out = outputs[2] 
        
        # 提取车道线掩码 (兼容单通道和双通道的 ONNX 导出格式)
        if lane_seg_out.shape[1] == 2:
            lane_mask_640 = lane_seg_out[0, 1, :, :] > lane_seg_out[0, 0, :, :]
        else:
            lane_mask_640 = lane_seg_out[0, 0, :, :] > 0.5
            
        # 3. 后处理：裁去 Padding，恢复并放大回原图尺寸 1280x720
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        lane_mask_cropped = lane_mask_640[top:640-bottom, left:640-right]
        lane_mask_original_size = cv2.resize(lane_mask_cropped.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

        # 4. 应用 p63 裁剪
        lane_mask_final = cv2.bitwise_and(lane_mask_original_size, lane_mask_original_size, mask=roi_mask)

        # 5. 绘制红色车道线到底图上
        frame[lane_mask_final == 1] = [0, 0, 255]

        # ==========================================
        # 模块 B：YOLO11 + ByteTrack 车辆检测 (已完成部分)
        # ==========================================
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=vehicle_classes, verbose=False)
        result = results[0]
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = box
                cls_name = yolo_model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} ID:{track_id} Conf:{conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20 - text_height), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")

    cap.release()
    out.release()
    print("视频处理完成，已保存至:", output_video_path)

if __name__ == "__main__":
    main()