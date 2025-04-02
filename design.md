# SiamFC Tracking ModelBox Design Document

## 1. 系统概述

本文档描述了基于ModelBox框架实现SiamFC单目标跟踪的设计方案。该方案将SiamFC跟踪器解耦为backbone和head两个独立组件,并集成到ModelBox的流水线架构中,实现高效的单目标跟踪功能。

## 2. FlowUnit设计

### 2.1 预处理FlowUnit (PreprocessingFlowUnit)

**功能描述**:
- 图像格式转换
- 图像预处理
- 数据格式标准化

**输入输出**:
- 输入:
  - `in_image`: 原始图像数据 (BGR格式)
  - `in_meta`: 图像元数据 (可选)
- 输出:
  - `out_image`: 预处理后的图像数据 (RGB格式)
  - `out_meta`: 处理后的元数据 (可选)

**处理流程**:
1. 图像解码
2. BGR到RGB转换
3. 图像尺寸调整(可选)
4. 数据格式标准化

### 2.2 Backbone特征提取FlowUnit (BackboneFlowUnit)

**功能描述**:
- AlexNet特征提取
- 特征图生成
- 特征标准化

**输入输出**:
- 输入:
  - `in_image`: 预处理后的图像数据 (RGB格式)
  - `in_meta`: 特征提取配置参数 (可选)
- 输出:
  - `out_feature`: 特征图数据
  - `out_meta`: 特征图元数据 (尺寸、通道数等)

**处理流程**:
1. 图像特征提取
2. 特征图标准化
3. 特征图缓存(用于模板特征)

### 2.3 模板特征管理FlowUnit (TemplateFlowUnit)

**功能描述**:
- 模板特征存储
- 特征更新管理
- 特征状态维护
- 视频流特征缓存管理

**输入输出**:
- 输入:
  - `in_feature`: 特征图数据
  - `in_meta`: 模板更新控制参数
    - `frame_id`: 帧ID
    - `is_first_frame`: 是否首帧
    - `target_box`: 目标框信息
    - `update_strategy`: 更新策略
- 输出:
  - `out_template`: 当前模板特征
  - `out_meta`: 模板状态信息
    - `template_age`: 模板年龄
    - `update_count`: 更新次数
    - `confidence_score`: 置信度分数

**处理流程**:
1. 特征存储
   - 首帧特征存储
   - 特征图缓存管理
   - 特征状态初始化

2. 状态更新
   - 基于置信度的更新策略
   - 基于时间间隔的更新策略
   - 基于目标变化的更新策略

3. 特征维护
   - 特征缓存清理
   - 特征质量评估
   - 特征版本管理

**更新策略**:
1. 基于置信度的更新
   ```python
   if confidence_score > confidence_threshold:
       # 更新模板特征
       update_template()
   ```

2. 基于时间间隔的更新
   ```python
   if frame_id - last_update_frame > update_interval:
       # 强制更新模板特征
       force_update_template()
   ```

3. 基于目标变化的更新
   ```python
   if target_box_change > change_threshold:
       # 更新模板特征
       update_template()
   ```

4. 混合更新策略
   ```python
   if (confidence_score > confidence_threshold and 
       frame_id - last_update_frame > update_interval):
       # 更新模板特征
       update_template()
   ```

### 2.4 Head跟踪FlowUnit (HeadFlowUnit)

**功能描述**:
- 快速交叉相关计算
- 响应图生成
- 目标定位

**输入输出**:
- 输入:
  - `in_template`: 模板特征
  - `in_feature`: 搜索图像特征
  - `in_meta`: 跟踪参数配置
- 输出:
  - `out_response`: 响应图
  - `out_meta`: 跟踪状态信息

**处理流程**:
1. 特征匹配
2. 响应图计算
3. 目标定位

### 2.5 后处理FlowUnit (PostprocessingFlowUnit)

**功能描述**:
- 响应图后处理
- 视频流结果处理
- 结果可视化
- 数据格式化
- 结果输出

**输入输出**:
- 输入:
  - `in_image`: 原始图像数据 (BGR格式)
  - `in_response`: 跟踪响应图
  - `in_meta`: 跟踪状态信息
    - `frame_id`: 帧ID
    - `timestamp`: 时间戳
    - `tracking_state`: 跟踪状态
- 输出:
  - `out_image`: 可视化结果图像
  - `out_data`: 格式化的跟踪结果数据
    - `tracking_results`: 跟踪结果列表
    - `performance_metrics`: 性能指标
    - `debug_info`: 调试信息

**处理流程**:
1. 响应图后处理
   - 响应图平滑
   - 峰值检测
   - 尺度估计
   - 目标框生成

2. 视频流处理
   - 帧率控制
   - 结果缓存
   - 平滑处理
   - 轨迹生成

3. 结果可视化
   - 目标框绘制
   - 轨迹绘制
   - 状态信息显示
   - 性能指标显示

4. 数据格式化
   - JSON格式转换
   - 数据压缩
   - 元数据添加
   - 结果验证

**视频流处理策略**:
1. 帧率控制
   ```python
   if frame_id - last_process_frame < min_frame_interval:
       # 跳过处理
       return
   ```

2. 结果平滑
   ```python
   # 使用滑动窗口平滑
   smoothed_box = smooth_trajectory(box_history, window_size)
   ```

3. 轨迹生成
   ```python
   # 维护轨迹历史
   trajectory = update_trajectory(box_history, max_length)
   ```

4. 性能优化
   ```python
   # 异步处理
   async def process_frame():
       # 处理当前帧
       result = await process_current_frame()
       # 更新显示
       await update_display(result)
   ```

**输出格式**:
```json
{
    "frame_id": 123,
    "timestamp": "2024-03-20T10:30:00.000Z",
    "tracking_results": {
        "box": [x, y, w, h],
        "confidence": 0.95,
        "scale": 1.0
    },
    "performance_metrics": {
        "fps": 30,
        "latency": 33.3,
        "memory_usage": 1024
    },
    "debug_info": {
        "template_age": 100,
        "update_count": 5,
        "tracking_state": "normal"
    }
}
```

## 3. Graph设计

**Graph配置**:
```toml
[graph]
name = "siamfc_tracking"
description = "SiamFC tracking with template management"

# 节点定义
[[graph.nodes]]
name = "video_input"
type = "input"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000

[[graph.nodes]]
name = "videodemuxer"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    format = "auto"
}

[[graph.nodes]]
name = "videodecoder"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    codec = "auto"
}

[[graph.nodes]]
name = "resize"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    width = 640,
    height = 480
}

[[graph.nodes]]
name = "normalize"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
}

[[graph.nodes]]
name = "frame_feature_extraction"
type = "flowunit"
device = "cuda"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    model_path = "pretrained/siamfc_alexnet_e50.pth",
    output_stride = 8
}

[[graph.nodes]]
name = "template_condition"
type = "condition"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    condition = "use_current_bbox"
}

[[graph.nodes]]
name = "extract_current_template"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    crop_size = [127, 127]
}

[[graph.nodes]]
name = "template_manager"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    max_template_age = 100,
    update_interval = 10
}

[[graph.nodes]]
name = "template_preprocessing"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    normalize = true
}

[[graph.nodes]]
name = "template_feature_extraction"
type = "flowunit"
device = "cuda"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    model_path = "pretrained/siamfc_alexnet_e50.pth"
}

[[graph.nodes]]
name = "template_expand"
type = "expand"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    num_candidates = 5,
    expand_ratio = 1.2
}

[[graph.nodes]]
name = "correlation_layer"
type = "flowunit"
device = "cuda"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    out_scale = 0.001
}

[[graph.nodes]]
name = "score_sorting"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    sort_by = "confidence"
}

[[graph.nodes]]
name = "select_best_bbox"
type = "flowunit"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000
config = {
    iou_threshold = 0.3,
    confidence_threshold = 0.5
}

[[graph.nodes]]
name = "output"
type = "output"
device = "cpu"
deviceid = 0
queue_size = 32
batch_size = 1
timeout = 1000

# 节点连接关系
[graph.connections]
# 视频输入处理
video_input -> videodemuxer
videodemuxer -> videodecoder
videodecoder -> resize
resize -> normalize

# 特征提取
normalize -> frame_feature_extraction

# 模板管理
normalize -> template_condition
template_condition -> extract_current_template
template_condition -> template_manager
template_manager -> template_preprocessing
extract_current_template -> template_preprocessing

# 模板特征提取
template_preprocessing -> template_feature_extraction
template_feature_extraction -> template_expand

# 相关计算
template_expand -> correlation_layer
frame_feature_extraction -> correlation_layer

# 结果处理
correlation_layer -> score_sorting
score_sorting -> select_best_bbox
select_best_bbox -> output

# 性能配置
[graph.performance]
enable_cuda_graph = true
enable_amp = true
enable_fusion = true
memory_pool_size = 1024

# 并行配置
[graph.parallel]
max_parallel = 5
batch_size = 1
timeout = 1000
thread_num = 4
```

## 4. 配置设计

### 4.1 Backbone配置
```toml
[backbone]
# 模型配置
model_path = "pretrained/siamfc_alexnet_e50.pth"
output_stride = 8
feature_dim = 256

# 性能配置
batch_size = 1
num_workers = 4
```

### 4.2 Head配置
```toml
[head]
# 跟踪参数
out_scale = 0.001
exemplar_sz = 127
instance_sz = 255
context = 0.5

# 尺度估计参数
scale_num = 3
scale_step = 1.0375
scale_lr = 0.59
scale_penalty = 0.9745
window_influence = 0.176

# 性能配置
batch_size = 1
num_workers = 4
```

### 4.3 预处理配置
```toml
[preprocessing]
# 图像处理
resize_width = 640
resize_height = 480
normalize = true
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 性能配置
num_workers = 4
```

### 4.4 后处理配置
```toml
[postprocessing]
# 可视化配置
draw_box = true
box_color = [0, 255, 0]
box_thickness = 2
draw_score = true

# 输出配置
output_format = "json"
save_image = true
```

### 4.5 目标检测配置
```toml
[detection]
# 模型配置
model_path = "pretrained/yolov5s.pt"
conf_threshold = 0.5
iou_threshold = 0.45

# 类别配置
classes = ["person", "car", "truck", "bus", "motorcycle"]
min_size = [20, 20]  # 最小目标尺寸

# 性能配置
batch_size = 1
num_workers = 4
```

### 4.6 结果融合配置
```toml
[fusion]
# 融合策略
strategy = "adaptive"  # adaptive, weighted, selective
iou_threshold = 0.3
confidence_threshold = 0.7

# 权重配置
track_weight = 0.7
detect_weight = 0.3

# 自适应参数
min_track_confidence = 0.6
max_detect_confidence = 0.9
```

## 5. 性能优化设计

### 5.1 计算优化
- 使用ModelBox的异步处理机制
- GPU加速特征提取和跟踪
- 批处理模式支持
- 内存池优化

### 5.2 内存优化
- 使用ModelBox内存池
- 零拷贝数据传输
- 特征图复用机制
- 模板特征缓存优化

### 5.3 并行优化
- 多线程预处理
- 流水线并行处理
- GPU并行计算
- 特征提取和跟踪并行

## 6. 错误处理设计

### 6.1 异常类型
- 输入数据异常
- 模型加载异常
- 特征提取异常
- 跟踪失败异常
- 系统资源异常

### 6.2 处理策略
- 优雅降级
- 错误恢复
- 状态保持
- 日志记录

## 7. 扩展性设计

### 7.1 模型扩展
- 支持不同版本的Backbone网络
- 支持不同版本的Head网络
- 支持模型热更新
- 支持模型切换

### 7.2 功能扩展
- 支持多目标跟踪
- 支持目标重识别
- 支持跟踪器切换
- 支持特征融合

## 8. 部署要求

### 8.1 硬件要求
- CPU: 4核以上
- 内存: 8GB以上
- GPU: NVIDIA GPU with CUDA support (推荐)

### 8.2 软件要求
- ModelBox >= 2.0.0
- CUDA >= 10.0
- PyTorch >= 1.7.0
- OpenCV >= 4.5.0

## 9. 测试策略

### 9.1 单元测试
- FlowUnit功能测试
- 配置参数测试
- 异常处理测试
- 特征提取测试
- 跟踪性能测试

### 9.2 性能测试
- 吞吐量测试
- 延迟测试
- 资源占用测试
- 特征提取性能测试
- 跟踪性能测试

### 9.3 集成测试
- 端到端功能测试
- 稳定性测试
- 压力测试
- 多目标场景测试

## 5. FlowUnit设计补充

### 5.1 目标检测FlowUnit (DetectionFlowUnit)

**功能描述**:
- 人和车辆目标检测
- 检测结果过滤
- 检测框优化

**输入输出**:
- 输入:
  - `in_image`: 预处理后的图像数据 (RGB格式)
  - `in_meta`: 检测配置参数
- 输出:
  - `out_boxes`: 检测框列表
    - `box`: [x, y, w, h]
    - `class`: 类别ID
    - `confidence`: 置信度
    - `class_name`: 类别名称
  - `out_meta`: 检测元数据
    - `detection_time`: 检测耗时
    - `num_detections`: 检测数量

**处理流程**:
1. 目标检测
   - 模型推理
   - NMS处理
   - 置信度过滤

2. 结果过滤
   - 类别过滤
   - 尺寸过滤
   - 置信度过滤

3. 检测框优化
   - 边界框平滑
   - 尺寸约束
   - 位置修正

### 5.2 跟踪结果融合FlowUnit (FusionFlowUnit)

**功能描述**:
- 跟踪和检测结果融合
- 自适应权重计算
- 结果优化

**输入输出**:
- 输入:
  - `in_track`: 跟踪结果
    - `box`: 跟踪框
    - `confidence`: 跟踪置信度
    - `track_id`: 跟踪ID
  - `in_detect`: 检测结果
    - `boxes`: 检测框列表
    - `classes`: 类别列表
    - `confidences`: 置信度列表
  - `in_meta`: 融合配置参数
- 输出:
  - `out_box`: 融合后的目标框
  - `out_meta`: 融合元数据
    - `fusion_strategy`: 使用的融合策略
    - `track_weight`: 跟踪权重
    - `detect_weight`: 检测权重
    - `final_confidence`: 最终置信度

**融合策略**:
1. 自适应融合
   ```python
   def adaptive_fusion(track_box, track_conf, detect_boxes, detect_confs):
       # 计算IoU
       ious = calculate_iou(track_box, detect_boxes)
       
       # 自适应权重计算
       if track_conf > min_track_confidence:
           track_weight = track_conf
           detect_weight = 1 - track_weight
       else:
           track_weight = 0.5
           detect_weight = 0.5
           
       # 加权融合
       fused_box = weighted_fusion(track_box, detect_boxes, 
                                 track_weight, detect_weight)
       return fused_box
   ```

2. 选择性融合
   ```python
   def selective_fusion(track_box, track_conf, detect_boxes, detect_confs):
       # 找到最佳匹配的检测框
       best_match = find_best_match(track_box, detect_boxes)
       
       if best_match.iou > iou_threshold:
           # 使用检测框
           return best_match.box
       else:
           # 使用跟踪框
           return track_box
   ```

3. 加权融合
   ```python
   def weighted_fusion(track_box, track_conf, detect_boxes, detect_confs):
       # 固定权重融合
       fused_box = (track_box * track_weight + 
                   detect_box * detect_weight)
       return fused_box
   ```

**结果优化**:
1. 置信度优化
   ```python
   def optimize_confidence(track_conf, detect_conf, iou):
       # 基于IoU的置信度调整
       if iou > 0.7:
           return max(track_conf, detect_conf)
       else:
           return (track_conf + detect_conf) / 2
   ```

2. 位置优化
   ```python
   def optimize_position(track_box, detect_box, track_conf, detect_conf):
       # 基于置信度的位置优化
       if track_conf > detect_conf:
           return track_box
       else:
           return detect_box
   ```

3. 尺寸优化
   ```python
   def optimize_size(box, min_size, max_size):
       # 尺寸约束
       w, h = box[2], box[3]
       w = max(min_size[0], min(w, max_size[0]))
       h = max(min_size[1], min(h, max_size[1]))
       return [box[0], box[1], w, h]
   ```

## 6. 性能优化设计补充

### 6.1 检测优化
- 检测模型量化
- 检测结果缓存
- 异步检测处理
- 检测框预测

### 6.2 融合优化
- 并行融合处理
- 结果缓存复用
- 自适应计算资源分配
- 融合策略动态调整

## 10. SiamFC特征管理优化

### 10.1 特征提取架构优化

**功能描述**:
- 分离Backbone和Head特征提取
- 支持多尺度特征处理
- 特征标准化和缓存

**实现细节**:
```python
class BackboneFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.backbone = AlexNetV1()
        self.feature_cache = {}
        
    def process(self, data_context):
        # 多尺度特征提取
        features = []
        for scale in scales:
            feature = self.backbone(scale_image)
            features.append(feature)
            # 特征缓存
            self.feature_cache[scale] = feature
        return features

class HeadFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.head = SiamFC(out_scale=0.001)
        
    def process(self, data_context):
        # 多尺度特征匹配
        responses = []
        for feature in features:
            response = self.head(self.kernel, feature)
            responses.append(response)
        return responses
```

### 10.2 模板特征管理优化

**功能描述**:
- 模板特征存储和更新
- 多尺度模板管理
- 特征质量评估

**实现细节**:
```python
class TemplateFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.kernel = None  # 模板特征
        self.center = None  # 目标中心
        self.target_sz = None  # 目标尺寸
        self.z_sz = None  # 模板尺寸
        self.avg_color = None  # 平均颜色
        self.feature_history = []  # 特征历史
        
    def process(self, data_context):
        if self.kernel is None:  # 首帧初始化
            # 提取和存储模板特征
            self.kernel = self.extract_template(in_feature)
            self.feature_history.append(self.kernel)
        else:  # 更新模板
            # 根据更新策略决定是否更新
            if self.should_update():
                new_kernel = self.extract_template(in_feature)
                self.kernel = self.update_template(new_kernel)
                self.feature_history.append(self.kernel)
                
    def should_update(self):
        # 基于置信度的更新
        if self.confidence > self.confidence_threshold:
            return True
            
        # 基于时间间隔的更新
        if self.frame_id - self.last_update > self.update_interval:
            return True
            
        # 基于目标变化的更新
        if self.target_change > self.change_threshold:
            return True
            
        return False
        
    def update_template(self, new_kernel):
        # 特征平滑更新
        alpha = self.update_rate
        return alpha * new_kernel + (1 - alpha) * self.kernel
```

### 10.3 搜索特征处理优化

**功能描述**:
- 多尺度搜索区域处理
- 特征图缓存
- 并行特征提取

**实现细节**:
```python
class SearchFeatureFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.scale_factors = None
        self.feature_cache = {}
        
    def process(self, data_context):
        # 多尺度搜索图像处理
        features = []
        for scale in self.scale_factors:
            # 检查缓存
            if scale in self.feature_cache:
                feature = self.feature_cache[scale]
            else:
                # 提取新特征
                feature = self.extract_feature(scale)
                self.feature_cache[scale] = feature
            features.append(feature)
            
        # 清理过期缓存
        self.clean_cache()
        return features
        
    def clean_cache(self):
        # 保留最近N帧的特征
        max_cache_size = 5
        if len(self.feature_cache) > max_cache_size:
            oldest_scale = min(self.feature_cache.keys())
            del self.feature_cache[oldest_scale]
```

### 10.4 特征匹配优化

**功能描述**:
- 快速交叉相关计算
- 响应图优化
- 多尺度融合

**实现细节**:
```python
class FeatureMatchingFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.hann_window = None
        self.response_cache = {}
        
    def process(self, data_context):
        # 多尺度响应计算
        responses = []
        for feature in features:
            response = self.compute_response(self.kernel, feature)
            responses.append(response)
            
        # 响应图优化
        optimized_response = self.optimize_responses(responses)
        return optimized_response
        
    def optimize_responses(self, responses):
        # 响应图平滑
        smoothed = []
        for response in responses:
            smoothed.append(self.smooth_response(response))
            
        # 尺度惩罚
        penalized = self.apply_scale_penalty(smoothed)
        
        # 峰值检测
        return self.find_peak(penalized)
```

### 10.5 性能优化策略

**内存优化**:
```python
class MemoryOptimizer:
    def __init__(self):
        self.memory_pool = {}
        self.max_pool_size = 100
        
    def allocate(self, size):
        if size in self.memory_pool:
            return self.memory_pool[size]
        return torch.empty(size)
        
    def release(self, tensor):
        # 回收内存
        del tensor
```

**计算优化**:
```python
class ComputeOptimizer:
    def __init__(self):
        self.batch_size = 1
        self.num_workers = 4
        
    def parallel_process(self, features):
        # 并行特征提取
        with torch.cuda.amp.autocast():
            results = []
            for feature in features:
                result = self.process_feature(feature)
                results.append(result)
        return results
```

**缓存优化**:
```python
class CacheOptimizer:
    def __init__(self):
        self.feature_cache = {}
        self.response_cache = {}
        
    def update_cache(self, key, value):
        # LRU缓存更新
        if len(self.feature_cache) > self.max_size:
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        self.feature_cache[key] = value
```

### 10.6 配置优化

```toml
[feature_management]
# 特征提取配置
feature_dim = 256
output_stride = 8
use_amp = true  # 自动混合精度

# 模板管理配置
max_template_age = 100
update_interval = 10
confidence_threshold = 0.7
change_threshold = 0.3

# 缓存配置
max_cache_size = 5
cache_cleanup_interval = 100

# 性能配置
batch_size = 1
num_workers = 4
use_cuda_graph = true  # CUDA图优化
```

### 10.7 多候选位置并行处理

**功能描述**:
- 基于预设框生成多个候选位置
- 并行特征提取和匹配
- 候选位置评分和筛选

**实现细节**:
```python
class CandidateExpandFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.num_candidates = 5  # 候选位置数量
        self.expand_ratio = 1.2  # 扩展比例
        
    def process(self, data_context):
        # 获取当前目标框
        current_box = data_context.get_input("current_box")
        
        # 生成候选位置
        candidates = self.generate_candidates(current_box)
        
        # 创建并行处理分支
        expand_data = []
        for candidate in candidates:
            expand_data.append({
                "candidate_box": candidate,
                "frame_id": data_context.get_input("frame_id"),
                "template_feature": data_context.get_input("template_feature")
            })
            
        # 输出到并行处理节点
        return expand_data
        
    def generate_candidates(self, current_box):
        candidates = []
        # 中心位置候选
        candidates.append(current_box)
        
        # 尺度变化候选
        for scale in [0.9, 1.1]:
            scaled_box = self.scale_box(current_box, scale)
            candidates.append(scaled_box)
            
        # 位置偏移候选
        for offset in [(5,0), (-5,0), (0,5), (0,-5)]:
            offset_box = self.offset_box(current_box, offset)
            candidates.append(offset_box)
            
        return candidates

class ParallelHeadFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.head = SiamFC(out_scale=0.001)
        self.response_threshold = 0.5
        
    def process(self, data_context):
        # 获取候选位置信息
        candidate_box = data_context.get_input("candidate_box")
        template_feature = data_context.get_input("template_feature")
        
        # 提取候选位置特征
        candidate_feature = self.extract_feature(candidate_box)
        
        # 计算相似度响应
        response = self.head(template_feature, candidate_feature)
        
        # 计算置信度分数
        confidence = self.compute_confidence(response)
        
        return {
            "candidate_box": candidate_box,
            "response": response,
            "confidence": confidence
        }

class CandidateMergeFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        self.iou_threshold = 0.3
        self.confidence_threshold = 0.5
        
    def process(self, data_context):
        # 获取所有候选结果
        candidate_results = data_context.get_input("candidate_results")
        
        # 按置信度排序
        sorted_results = sorted(
            candidate_results,
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        # 非极大值抑制
        selected_results = self.nms(sorted_results)
        
        # 选择最佳结果
        best_result = selected_results[0]
        
        return {
            "final_box": best_result["candidate_box"],
            "confidence": best_result["confidence"],
            "response": best_result["response"]
        }
        
    def nms(self, results):
        selected = []
        while results:
            best = results.pop(0)
            selected.append(best)
            
            # 移除重叠的候选框
            results = [
                r for r in results
                if self.iou(best["candidate_box"], r["candidate_box"]) < self.iou_threshold
            ]
        return selected
```

**Graph配置**:
```toml
[graph]
# 节点配置
nodes = [
    {name = "candidate_expand", type = "expand", input = ["current_box", "frame_id", "template_feature"]},
    {name = "parallel_head", type = "flowunit", input = ["candidate_box", "frame_id", "template_feature"]},
    {name = "candidate_merge", type = "merge", input = ["candidate_results"]}
]

# 边配置
edges = [
    {from = "candidate_expand", to = "parallel_head"},
    {from = "parallel_head", to = "candidate_merge"}
]

# 并行配置
parallel_config = {
    max_parallel = 5,
    batch_size = 1,
    timeout = 1000
}
```

**性能优化**:
```python
class ParallelOptimizer:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        self.feature_cache = {}
        
    def parallel_process(self, candidates):
        # 并行处理候选位置
        futures = []
        for candidate in candidates:
            future = self.thread_pool.submit(
                self.process_candidate,
                candidate
            )
            futures.append(future)
            
        # 等待所有结果
        results = [f.result() for f in futures]
        return results
        
    def process_candidate(self, candidate):
        # 检查特征缓存
        cache_key = self.get_cache_key(candidate)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        # 处理新候选位置
        result = self.extract_and_match(candidate)
        self.feature_cache[cache_key] = result
        return result
```

**配置优化**:
```toml
[candidate_processing]
# 候选生成配置
num_candidates = 5
expand_ratio = 1.2
scale_factors = [0.9, 1.1]
offset_pixels = [5, -5]

# 并行处理配置
max_parallel = 5
batch_size = 1
timeout = 1000

# 筛选配置
iou_threshold = 0.3
confidence_threshold = 0.5

# 缓存配置
max_cache_size = 100
cache_cleanup_interval = 1000
``` 


Original Frame
    ↓
3 Different Sized Crops:
    - 0.964x size crop
    - 1.000x size crop
    - 1.0375x size crop
    ↓
All Resized to 255x255
    ↓
Processed through network
    ↓
3 Response Maps
    ↓
Best scale selected based on maximum response