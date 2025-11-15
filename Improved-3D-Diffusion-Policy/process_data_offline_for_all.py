# 处理离线数据的代码
# 根据离线数据集生成mask

import copy
import os
import sys 
import tempfile
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
import argparse
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
import gc

# 1. 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录 (.../level2/)
current_dir = os.path.dirname(current_file_path)
# 3. 获取上级目录 (.../level1/)
project_root = os.path.dirname(current_dir)
# 4. 添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class DDSGroundingDinoPredictor:
    """
    Wrapper for using DDS Cloud API (GroundingDINO 1.5) for detection.
    """
    def __init__(self, api_token: str, model_name="GroundingDino-1.5-Pro", device="cuda"):
        """
        Initialize the DDS Cloud API predictor.
        Args:
            api_token (str): Your DDS API token.
            model_name (str): The model to use (e.g., "GroundingDino-1.5-Pro").
            device (str): Device to move output tensors to.
        """
        self.model_name = model_name
        self.config = Config(api_token)
        self.client = Client(self.config)
        self.device = device
        print(f"[DDSGroundingDinoPredictor] Initialized with model: {self.model_name}")

    def predict(
        self,
        image: "PIL.Image.Image",
        text_prompts: str,
        box_threshold=0.2,
        text_threshold=0.2, # Note: text_threshold is aliased to iou_threshold here
    ):
        """
        Perform object detection using the DDS Cloud API.
        Args:
            image (PIL.Image.Image): Input RGB image.
            text_prompts (str): Text prompt describing target objects.
            box_threshold (float): Confidence threshold for box selection.
            text_threshold (float): Mapped to iou_threshold for the API.
        Returns:
            Tuple[Tensor, List[str]]: Bounding boxes and matched class labels.
        """
        print("[DDS] Preparing image for upload...")
        # 1. Convert PIL Image to np array (BGR for cv2)
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # 2. Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_filename = tmpfile.name
            cv2.imwrite(temp_filename, image_np_bgr)

        try:
            # 3. Upload file
            print(f"[DDS] Uploading {temp_filename}...")
            image_url = self.client.upload_file(temp_filename)
            print(f"[DDS] Image uploaded, URL: {image_url}")

            # 4. Define and run the task
            task = V2Task(
                api_path="/v2/task/grounding_dino/detection",
                api_body={
                    "model": self.model_name,
                    "image": image_url,
                    "prompt": {
                        "type": "text",
                        "text": text_prompts
                    },
                    "targets": ["bbox"],
                    "bbox_threshold": box_threshold,
                    "iou_threshold": text_threshold, # Using text_threshold as IOU
                }
            )
            print("[DDS] Running detection task... (This may take a moment)")
            self.client.run_task(task)
            result = task.result
            print("[DDS] Task complete, processing results.")

            # 5. Process results
            objects = result.get("objects", [])
            input_boxes = []
            labels = []

            if not objects:
                print("[DDS] No objects found.")
                return torch.empty((0, 4), device=self.device), []

            for obj in objects:
                input_boxes.append(obj["bbox"])
                labels.append(obj["category"].lower().strip())
            
            # Convert to tensor and move to device
            boxes_tensor = torch.tensor(np.array(input_boxes).reshape(-1, 4), dtype=torch.float32).to(self.device)
            return boxes_tensor, labels

        finally:
            # 6. Clean up temporary file
            os.remove(temp_filename)
            print(f"[DDS] Cleaned up temp file {temp_filename}.")

class SAM2ImageSegmentor:
    """
    Wrapper class for SAM2-based segmentation given bounding boxes.
    """

    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device="cuda"):
        """
        Initialize the SAM2 image segmentor.
        Args:
            sam_model_cfg (str): Path to the SAM2 config file.
            sam_model_ckpt (str): Path to the SAM2 checkpoint file.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.
        Args:
            image (np.ndarray): RGB image array with shape (H, W, 3).
        """
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        """
        Predict segmentation masks from given bounding boxes.
        Args:
            boxes (torch.Tensor): Bounding boxes as (N, 4) tensor.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - masks: Binary masks per box, shape (N, H, W)
                - scores: Confidence scores for each mask
                - logits: Raw logits from the model
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # Normalize shape to (N, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits


class IncrementalObjectTracker:
    def __init__(
        self,
        api_token: str,
        grounding_model_name: str = "GroundingDino-1.5-Pro",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text="car.",
        detection_interval=20,
    ):
        self.device = device
        self.detection_interval = detection_interval
        self.prompt_text = prompt_text

        # Load models
        self.grounding_predictor = DDSGroundingDinoPredictor(
            api_token=api_token, model_name=grounding_model_name, device=device
        )
        self.sam2_segmentor = SAM2ImageSegmentor(
            sam_model_cfg=sam2_model_cfg,
            sam_model_ckpt=sam2_ckpt_path,
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(
            sam2_model_cfg, sam2_ckpt_path
        )

        # Initialize state
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        self.total_frames = 0
        self.objects_count = 0
        self.frame_cache_limit = 40

        self.last_mask_dict = MaskDictionaryModel()
        self.track_dict = MaskDictionaryModel()

        # ✅ new fields for memory reset without losing masks
        self.next_anchor_masks = None
        self.need_reset_next_frame = False

    # ============================================================
    # ✅ add_frame: handles cache and re-anchor logic
    # ============================================================
    def add_frame(self, image_np):
        """
        改动目标：
            - 不做重锚定（不重新 add_new_mask）
            - 仅在缓存过多时清理 inference_state 中的旧帧数据
            - 保持跟踪连续性
        """
        frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)

        # 获取当前缓存帧数
        frames_cached = self.inference_state["images"].shape[0]

        if frames_cached > self.frame_cache_limit:
            keep_n = self.frame_cache_limit // 2
            for key in ["images", "features", "embeddings"]:
                if key in self.inference_state and isinstance(self.inference_state[key], torch.Tensor):
                    if self.inference_state[key].shape[0] > keep_n:
                        self.inference_state[key] = self.inference_state[key][-keep_n:].clone()

            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Memory Prune] Cached frames pruned to last {keep_n} entries")

        return frame_idx



    # ============================================================
    # ✅ modified add_image -> uses add_frame
    # ============================================================
    def add_image(self, image_np: np.ndarray):

        img_pil = Image.fromarray(image_np)

        # 初始化视频尺寸
        if self.inference_state["video_height"] is None:
            self.inference_state["video_height"], self.inference_state["video_width"] = image_np.shape[:2]

        # --------------------------------------------------------
        # A) 检测帧
        # --------------------------------------------------------
        if self.total_frames % self.detection_interval == 0:
            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)

            if boxes.shape[0] == 0:
                # 无检测结果 → 直接通过传播
                frame_idx = self.add_frame(image_np)
            else:
                with torch.no_grad():
                    self.sam2_segmentor.set_image(image_np)
                    masks, scores, logits = self.sam2_segmentor.predict_masks_from_boxes(boxes)

                mask_dict = MaskDictionaryModel(
                    promote_type="mask",
                    mask_name=f"mask_{self.total_frames:05d}.npy"
                )
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(self.device),
                    box_list=torch.tensor(boxes),
                    label_list=labels,
                )

                self.objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=self.last_mask_dict,
                    iou_threshold=0.3,
                    objects_count=self.objects_count,
                )

                frame_idx = self.add_frame(image_np)
                self.video_predictor.reset_state(self.inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, _, _ = self.video_predictor.add_new_mask(
                        self.inference_state,
                        frame_idx,
                        object_id,
                        object_info.mask,
                    )

                # 不重锚定，不 reset_state，直接保留现有跟踪状态
                self.track_dict = copy.deepcopy(mask_dict)
                self.last_mask_dict = mask_dict

        # --------------------------------------------------------
        # B) 非检测帧 → 正常增量跟踪
        # --------------------------------------------------------
        else:
            frame_idx = self.add_frame(image_np)

        # --------------------------------------------------------
        # C) 推理传播当前帧掩码
        # --------------------------------------------------------
        with torch.no_grad():
            frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
            )

        frame_masks = MaskDictionaryModel()
        for i, obj_id in enumerate(obj_ids):
            out_mask = video_res_masks[i] > 0.0
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=self.track_dict.get_target_class_name(obj_id),
                logit=self.track_dict.get_target_logit(obj_id),
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
            frame_masks.mask_name = f"mask_{self.total_frames + 1:05d}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        self.last_mask_dict = copy.deepcopy(frame_masks)

        # 生成可视化帧
        H, W = image_np.shape[:2]
        mask_img = torch.zeros((H, W), dtype=torch.int32)
        for obj_id, obj_info in self.last_mask_dict.labels.items():
            mask_img[obj_info.mask] = obj_id
        mask_array = mask_img.cpu().numpy()

        annotated_frame = self.visualize_frame_with_mask_and_metadata(
            image_np=image_np,
            mask_array=mask_array,
            json_metadata=self.last_mask_dict.to_dict(),
        )

        self.total_frames += 1
        if self.total_frames % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # print(f"[Tracker] Total processed frames: {self.total_frames}")
        return annotated_frame, mask_array



    def set_prompt(self, new_prompt: str):
        """
        Dynamically update the GroundingDINO prompt and reset tracking state
        to force a new object detection.
        """
        self.prompt_text = new_prompt
        self.total_frames = 0  # Trigger immediate re-detection
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty(
            (0, 3, 1024, 1024), device=self.device
        )
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        print(f"[Prompt Updated] New prompt: '{new_prompt}'. Tracker state reset.")

    def save_current_state(self, output_dir, raw_image: np.ndarray = None):
        """
        Save the current mask, metadata, raw image, and annotated result.
        Args:
            output_dir (str): The root output directory.
            raw_image (np.ndarray, optional): The original input image (RGB).
        """
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        image_data_dir = os.path.join(output_dir, "images")
        vis_data_dir = os.path.join(output_dir, "result")

        os.makedirs(mask_data_dir, exist_ok=True)
        os.makedirs(json_data_dir, exist_ok=True)
        os.makedirs(image_data_dir, exist_ok=True)
        os.makedirs(vis_data_dir, exist_ok=True)

        frame_masks = self.last_mask_dict

        # Ensure mask_name is valid
        if not frame_masks.mask_name or not frame_masks.mask_name.endswith(".npy"):
            frame_masks.mask_name = f"mask_{self.total_frames:05d}.npy"

        base_name = f"image_{self.total_frames:05d}"

        # Save segmentation mask
        mask_img = torch.zeros(frame_masks.mask_height, frame_masks.mask_width)
        for obj_id, obj_info in frame_masks.labels.items():
            mask_img[obj_info.mask == True] = obj_id
        np.save(
            os.path.join(mask_data_dir, frame_masks.mask_name),
            mask_img.numpy().astype(np.uint16),
        )

        # Save metadata as JSON
        json_path = os.path.join(json_data_dir, base_name + ".json")
        frame_masks.to_json(json_path)

        # Save raw input image
        if raw_image is not None:
            image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_data_dir, base_name + ".jpg"), image_bgr)

            # Save annotated image with mask, bounding boxes, and labels
            annotated_image = self.visualize_frame_with_mask_and_metadata(
                image_np=raw_image,
                mask_array=mask_img.numpy().astype(np.uint16),
                json_metadata=frame_masks.to_dict(),
            )
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(vis_data_dir, base_name + "_annotated.jpg"), annotated_bgr
            )
            print(
                f"[Saved] {base_name}.jpg and {base_name}_annotated.jpg saved successfully."
            )

    def visualize_frame_with_mask_and_metadata(
        self,
        image_np: np.ndarray,
        mask_array: np.ndarray,
        json_metadata: dict,
    ):
        image = image_np.copy()
        H, W = image.shape[:2]

        # Step 1: Parse metadata and build object entries
        metadata_lookup = json_metadata.get("labels", {})

        all_object_ids = []
        all_object_boxes = []
        all_object_classes = []
        all_object_masks = []

        for obj_id_str, obj_info in metadata_lookup.items():
            instance_id = obj_info.get("instance_id")
            if instance_id is None or instance_id == 0:
                continue
            if instance_id not in np.unique(mask_array):
                continue

            object_mask = mask_array == instance_id
            all_object_ids.append(instance_id)
            x1 = obj_info.get("x1", 0)
            y1 = obj_info.get("y1", 0)
            x2 = obj_info.get("x2", 0)
            y2 = obj_info.get("y2", 0)
            all_object_boxes.append([x1, y1, x2, y2])
            all_object_classes.append(obj_info.get("class_name", "unknown"))
            all_object_masks.append(object_mask[None])  # Shape (1, H, W)

        # Step 2: Check if valid objects exist
        if len(all_object_ids) == 0:
            print("No valid object instances found in metadata.")
            return image

        # Step 3: Sort by instance ID
        paired = list(
            zip(all_object_ids, all_object_boxes, all_object_masks, all_object_classes)
        )
        paired.sort(key=lambda x: x[0])

        all_object_ids = [p[0] for p in paired]
        all_object_boxes = [p[1] for p in paired]
        all_object_masks = [p[2] for p in paired]
        all_object_classes = [p[3] for p in paired]

        # Step 4: Build detections
        all_object_masks = np.concatenate(all_object_masks, axis=0)
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_object_classes)
        ]

        # Step 5: Annotate image
        annotated_frame = image.copy()
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = mask_annotator.annotate(annotated_frame, detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame

# ... (IncrementalObjectTracker 类的定义结束)

import os
# ... (其他 import)
import pyrealsense2 as rs 
import numpy as np        
from pathlib import Path
import json
import time

def process_episode(episode_name, dataset_root_path, base_output_dir, api_token):
    """
    处理单个 episode 的所有逻辑。
    
    Args:
        episode_name (str): 要处理的 episode 目录名 (例如: "episode_0001")
        dataset_root_path (Path): 包含所有 episode 的根数据集路径
        base_output_dir (Path): 所有处理后数据的根输出路径
        api_token (str): DDS API 密钥
    """
    print(f"\n" + "="*50)
    print(f"▶️ 开始处理 Episode: {episode_name}")
    print("="*50)
    
    # --- 1. 参数设置 (从 main 移入) ---
    # 这些是每个 episode 特有的设置
    output_dir = base_output_dir / f"{episode_name}_processed"
    input_dataset_dir = dataset_root_path / episode_name
    
    # 这些是固定的配置
    prompt_text = "brown-cup. yellow-cup. white-basket."
    detection_interval = 1000
    max_frames = 10000
    
    os.makedirs(output_dir, exist_ok=True)
    
    depth_data_dir = os.path.join(output_dir, "depth_data")
    os.makedirs(depth_data_dir, exist_ok=True)

    # --- 2. 初始化 Tracker ---
    # Tracker 必须为每个 episode 重新初始化
    tracker = IncrementalObjectTracker(
        api_token=api_token, # <--- 使用传入的 token
        grounding_model_name="GroundingDino-1.5-Pro",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_tiny.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt(prompt_text)

    # --- 3. 加载并复制内参 ---
    input_intrinsics_path = os.path.join(input_dataset_dir, "intrinsics.json")
    output_intrinsics_path = os.path.join(output_dir, "intrinsics.json")
    
    input_aligned_data_path = os.path.join(input_dataset_dir, "aligned_data.json")
    output_aligned_data_path = os.path.join(output_dir, "aligned_data.json")

    # [修改] 如果文件不存在，打印错误并跳过此 episode
    if not os.path.exists(input_intrinsics_path):
        print(f"[Error] 'intrinsics.json' not found in {input_dataset_dir}. 跳过此 episode.")
        return # <--- 退出此函数，继续下一个 episode
    
    try:
        with open(input_intrinsics_path, 'r') as f:
            intrinsics_data = json.load(f)
        with open(output_intrinsics_path, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)
        print(f"[Info] Copied intrinsics to {output_intrinsics_path}")
    except Exception as e:
        print(f"[Error] Failed to read/write intrinsics: {e}. 跳过此 episode.")
        return # <--- 退出

    try:
        with open(input_aligned_data_path, 'r') as f:
            aligned_data = json.load(f)
        with open(output_aligned_data_path, 'w') as f:
            json.dump(aligned_data, f, indent=4)
        print(f"[Info] Copied aligned data to {output_aligned_data_path}")
    except Exception as e:
        print(f"[Error] Failed to read/write aligned data: {e}. 跳过此 episode.")
        return
        
    # --- 4. 加载数据集元数据 ---
    metadata_path = os.path.join(input_dataset_dir, "time.txt")
    if not os.path.exists(metadata_path):
        print(f"[Error] 'time.txt' not found in {input_dataset_dir}. 跳过此 episode.")
        return # <--- 退出

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read metadata '{metadata_path}': {e}. 跳过此 episode.")
        return
    
    frame_ids = sorted(metadata.keys())
    
    if not frame_ids:
        print("[Error] No frames found in metadata. 跳过此 episode.")
        return

    print(f"[Info] 找到 {len(frame_ids)} 帧待处理 (来自 {input_dataset_dir})")

    # --- 5. 主循环 ---
    print("[Info] 开始处理... 按 'q' 键跳过当前 episode.")
    frame_idx = 0
    prev_frame_time = time.time()
    
    # [修改] 为每个 episode 创建唯一的窗口名称
    window_title = f"Offline Inference [{episode_name}] (RGB Tracker | Depth Colormap)"

    try:
        for i, frame_id in enumerate(frame_ids):
            frame_info = metadata[frame_id]
            
            # 1. 构建文件路径
            rgb_filename = frame_info.get("rgb")
            depth_npy_filename = frame_info.get("depth_npy") 
            
            if not rgb_filename or not depth_npy_filename:
                print(f"[Warning] 跳过帧 {frame_id}: metadata 中缺少 'rgb' 或 'depth_npy'.")
                continue
                
            rgb_path = os.path.join(input_dataset_dir, "rgb", rgb_filename)
            depth_path = os.path.join(input_dataset_dir, "depth_npy", depth_npy_filename)

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"[Warning] 跳过帧 {frame_id}: 文件未找到. ({rgb_path} or {depth_path})")
                continue
            
            # 2. 从文件加载图像
            frame_bgr = cv2.imread(rgb_path)
            if frame_bgr is None:
                print(f"[Warning] 跳过帧 {frame_id}: 无法读取 RGB 图像 {rgb_path}")
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            depth_image_np = np.load(depth_path)
            if depth_image_np is None:
                 print(f"[Warning] 跳过帧 {frame_id}: 无法读取深度图像 {depth_path}")
                 continue

            # 3. FPS 计算
            new_frame_time = time.time()
            T_loop = new_frame_time - prev_frame_time
            prev_frame_time = new_frame_time
            processing_fps = 1.0 / T_loop
            print(f"[Frame {frame_idx} (File: {frame_id})] Processing... (Processing FPS: {processing_fps:.2f})")

            # 4. 使用 tracker 处理 RGB 帧
            process_image_rgb = tracker.add_image(frame_rgb)

            if process_image_rgb is None or not isinstance(process_image_rgb, np.ndarray):
                print(f"[Warning] 跳过帧 {frame_idx} (Tracker 返回空结果).")
                frame_idx += 1
                continue

            # 5. 可视化
            process_image_bgr = cv2.cvtColor(process_image_rgb, cv2.COLOR_RGB2BGR)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image_np, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            stacked_image = np.hstack((process_image_bgr, depth_colormap))
            
            cv2.imshow(window_title, stacked_image) # <--- 使用新窗口标题

            # [修改] 按 'q' 键会触发 KeyboardInterrupt，以便 finally 块能正确执行
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Info] 'q' 被按下，跳过当前 episode...")
                raise KeyboardInterrupt # <--- 触发中断
                
            # 6. 保存状态
            tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
            
            depth_filename = os.path.join(depth_data_dir, f"depth_{frame_idx + 1:05d}.npy")
            np.save(depth_filename, depth_image_np)

            frame_idx += 1
            if frame_idx >= max_frames:
                print(f"[Info] 达到 {max_frames} 帧上限. 停止当前 episode.")
                break
                
    except KeyboardInterrupt:
        print(f"[Info] 中断 {episode_name} 的处理。")
    finally:
        # [修改] 只关闭当前 episode 的窗口
        cv2.destroyWindow(window_title) 
        print(f"✅ 完成 Episode: {episode_name}")

#
# -----------------------------------------------------------------
#

def main():
    # --- 1. [修改] 定义根路径和共享参数 ---
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # (如果不需要，可以移除)
    # current_script_path = Path(__file__).resolve()
    # project_root = current_script_path.parent.parent.parent
    
    # **指定您的数据集根目录**
    dataset_root_path = Path("/media/shui/Lexar/put_cup_into_basket_1106_1600_1800")  
    # **指定您的总输出根目录**
    base_output_dir = dataset_root_path.parent / "put_cup_into_basket_1106_1600_1800_processed"
    
    API_TOKEN = "73ebeba4255b3cf2a3dc13368ac25191"

    # --- 2. [新增] 自动发现所有 Episodes ---
    if not os.path.isdir(dataset_root_path):
        print(f"[Fatal Error] 数据集根目录未找到: {dataset_root_path}")
        return

    episode_dirs = []
    for item in os.listdir(dataset_root_path):
        item_path = dataset_root_path / item
        # 检查它是否是一个目录 并且 名字是否以 "episode_" 开头
        if os.path.isdir(item_path) and item.startswith("episode_"):
            episode_dirs.append(item)
    
    episode_dirs.sort() # 确保按 "episode_0001", "episode_0002" ... 的顺序处理

    if not episode_dirs:
        print(f"[Warning] 在 {dataset_root_path} 中未找到 'episode_' 目录。")
        return

    print(f"[Info] 发现 {len(episode_dirs)} 个 Episodes: {episode_dirs}")

    # --- 3. [新增] 循环处理所有 Episodes ---
    try:
        for episode_name in episode_dirs:
            # 调用我们新创建的函数
            process_episode(
                episode_name=episode_name,
                dataset_root_path=dataset_root_path,
                base_output_dir=base_output_dir,
                api_token=API_TOKEN
            )
            
            # (可选) 在每个 episode 之间暂停一下，释放显存
            torch.cuda.empty_cache()
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\n[Info] 主进程被 (Ctrl+C) 中断。停止所有处理。")
    finally:
        cv2.destroyAllWindows()
        print("\n[Done] 所有 Episodes 处理完毕。")


if __name__ == "__main__":
    main()