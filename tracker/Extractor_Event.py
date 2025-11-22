import cv2
import numpy as np


class WhiteBgFeatureExtractor:
    def __init__(self, bg_color=(255, 255, 255), tolerance=0):
        self.bg_color = np.array(bg_color)  # 背景色阈值
        self.tolerance = tolerance  # 颜色容差

    def extract(self, image):
        """提取非背景像素的空间分布特征"""
        # 创建背景掩码（RGB精确匹配）
        bg_mask = np.all(np.abs(image - self.bg_color) <= self.tolerance, axis=2)
        y, x = np.where(~bg_mask)  # 非背景像素坐标

        if len(x) == 0:  # 处理全背景图像
            feature = np.zeros(7) + 1e-7
            return  feature

        # 计算空间分布特征
        features = [
            len(x),  # 非背景像素数量
            np.mean(x),  # 标准化X均值
            np.std(x),  # 标准化X标准差
            np.mean(y),  # 标准化Y均值
            np.std(y),  # 标准化Y标准差
            np.ptp(x),  # 标准化X跨度
            np.ptp(y),  # 标准化Y跨度
        ]
        return np.array(features)


class EventExtractor:
    def __init__(self, target_size=(64, 64)):
        self.extractor = WhiteBgFeatureExtractor(bg_color=(255, 255, 255))
        self.target_size = target_size
        self.feature_dim = 8

    def _preprocess(self, im_crops):
        """保持RGB通道的预处理"""
        processed = []
        for im in im_crops:
            # 统一尺寸并保持RGB格式
            resized = cv2.resize(im, self.target_size)
            if resized.shape[-1] != 3:  # 处理单通道伪彩色
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            processed.append(resized)
        return processed

    def __call__(self, im_crops):
        """返回标准化后的特征矩阵"""
        preprocessed = self._preprocess(im_crops)
        features = np.array([self.extractor.extract(im) for im in preprocessed])

        # 3. L2归一化 (按行归一化)
        norms = np.linalg.norm(features, axis=1, keepdims=True)

        norms[norms == 0] = 1  # 防止除以零
        normalized_features = features / norms
        return normalized_features.astype(np.float32)


if __name__ == "__main__":
    # 初始化提取器
    extractor = EventExtractor()

    # 测试图像加载（直接读取为RGB）
    test_images = [
        cv2.imread('/mnt/sda/Disk_D/Anti-UAV-VET-test/event_crops_for_rgb_50/target_284/frame0000.jpg'),
        cv2.imread('/mnt/sda/Disk_D/Anti-UAV-VET-test/event_crops_for_rgb_50/target_284/frame0900.jpg'),
    ]

    # 提取并验证特征
    features = extractor(test_images)

    for feature in features:
        print(feature)

    # 计算相似度（余弦相似度更适合标准化特征）
    cos_sim = np.dot(features[0], features[1]) / (np.linalg.norm(features[0]) * np.linalg.norm(features[1]))
    print(f"相似度: {cos_sim:.4f}")