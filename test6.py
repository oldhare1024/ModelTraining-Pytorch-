import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义字符集: 0-9 (10个) + a-z (26个) = 36个字符
CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]
NUM_CLASSES = len(CHARACTERS)

def create_alphanumeric_data(batch_size=1200, image_size=64):
    """创建字母数字混合的彩色图像数据"""
    images = []
    masks = []
    
    for _ in range(batch_size):
        # 随机选择字符
        char_idx = np.random.randint(0, NUM_CLASSES)
        char = CHARACTERS[char_idx]
        
        # 创建背景
        bg_color = np.random.randint(30, 180, 3) / 255.0
        background = np.ones((image_size, image_size, 3), dtype=np.float32) * bg_color.reshape(1, 1, 3)
        
        # 创建画布
        canvas_size = int(image_size * 1.5)
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        canvas_color = np.random.rand(3) * 0.5 + 0.2
        canvas = canvas * canvas_color.reshape(1, 1, 3)
        
        # 字体参数
        font_scale = np.random.uniform(1.3, 2.0)
        thickness = np.random.randint(2, 4)
        
        # 计算字符位置
        (text_width, text_height), _ = cv2.getTextSize(
            char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        center_x = canvas_size // 2
        center_y = canvas_size // 2
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        
        # 在画布上绘制字符
        text_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
        
        # 字符颜色
        text_color = np.random.randint(150, 255, 3) / 255.0
        
        # 20%概率让字符颜色与背景接近
        if np.random.random() < 0.2:
            text_color = bg_color + np.random.randn(3) * 0.08
            text_color = np.clip(text_color, 0.2, 0.9)
        
        cv2.putText(text_canvas, char, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   tuple(float(c) for c in text_color), thickness)
        
        # 将字符叠加到画布
        alpha = np.random.uniform(0.7, 0.95)
        text_mask = text_canvas.sum(axis=2) > 0.1
        
        for c in range(3):
            canvas[:, :, c][text_mask] = (
                alpha * text_canvas[:, :, c][text_mask] + 
                (1 - alpha) * canvas[:, :, c][text_mask]
            )
        
        # 裁剪字符区域
        start_x = center_x - image_size // 2
        start_y = center_y - image_size // 2
        char_region = canvas[start_y:start_y+image_size, start_x:start_x+image_size]
        
        # 创建二值掩码
        mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        cv2.putText(mask_canvas, char, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1.0, thickness)
        
        mask = mask_canvas[start_y:start_y+image_size, start_x:start_x+image_size]
        mask = (mask > 0.5).astype(np.float32)
        
        # 将字符混合到背景
        final_image = background.copy()
        char_mask = mask > 0
        
        blend_alpha = np.random.uniform(0.6, 0.85)
        for c in range(3):
            final_image[:, :, c][char_mask] = (
                blend_alpha * char_region[:, :, c][char_mask] + 
                (1 - blend_alpha) * background[:, :, c][char_mask]
            )
        
        # 添加噪声
        noise = np.random.randn(image_size, image_size, 3) * 0.03
        final_image = np.clip(final_image + noise, 0, 1)
        
        # 转换为PyTorch格式
        image_tensor = final_image.transpose(2, 0, 1)  # (3, H, W)
        mask_tensor = mask[np.newaxis, :, :]           # (1, H, W)
        
        images.append(image_tensor)
        masks.append(mask_tensor)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

class SimpleUNet(nn.Module):
    """简化的UNet模型"""
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.final = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # 中间层
        m = self.middle(self.pool2(e2))
        
        # 解码器
        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

def main():
    # 实验参数
    TOTAL_SIZE = 800
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 20
    
    print("生成数据...")
    images, masks = create_alphanumeric_data(batch_size=TOTAL_SIZE, image_size=IMAGE_SIZE)
    
    # 划分训练集和验证集
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {len(train_images)}张, 验证集: {len(val_images)}张")
    
    # 转换为PyTorch张量
    train_images_tensor = torch.FloatTensor(train_images)
    train_masks_tensor = torch.FloatTensor(train_masks)
    val_images_tensor = torch.FloatTensor(val_images)
    val_masks_tensor = torch.FloatTensor(val_masks)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleUNet().to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    print("开始训练...")
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        indices = np.random.permutation(len(train_images))
        
        for i in range(0, len(train_images), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            batch_input = train_images_tensor[batch_idx].to(device)
            batch_target = train_masks_tensor[batch_idx].to(device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(val_images), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(val_images))
                batch_input = val_images_tensor[i:batch_end].to(device)
                batch_target = val_masks_tensor[i:batch_end].to(device)
                
                output = model(batch_input)
                loss = criterion(output, batch_target)
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / (len(train_images) / BATCH_SIZE)
        avg_val_loss = val_loss / (len(val_images) / BATCH_SIZE)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{EPOCHS}: 训练Loss={avg_train_loss:.4f}, 验证Loss={avg_val_loss:.4f}")
    
    print("训练完成!")
    
    # 可视化结果
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        
        for i in range(5):
            idx = np.random.randint(len(val_images))
            
            img = val_images_tensor[idx:idx+1].to(device)
            mask = val_masks_tensor[idx:idx+1].to(device)
            pred = model(img)
            
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            mask_np = mask[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            
            # 原始图像
            axes[0, i].imshow(img_np)
            axes[0, i].set_title('原始图像')
            axes[0, i].axis('off')
            
            # 真实掩码
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title('真实掩码')
            axes[1, i].axis('off')
            
            # 预测掩码
            axes[2, i].imshow(pred_np, cmap='gray')
            axes[2, i].set_title('预测掩码')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 训练曲线
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(val_losses, label='验证损失', linewidth=2)
        plt.title('训练和验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return model

if __name__ == "__main__":
    model = main()
