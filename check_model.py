import torch
from safetensors.torch import load_file
try:
    from pi3.models.pi3 import Pi3
except Exception as e:
    print("Warning: failed to import Pi3. Make sure your PYTHONPATH includes the project. Error:", e)
    Pi3 = None
# === 1️⃣ 配置 ===
safetensor_path = "./ckpts/model.safetensors"  # ✅ 修改成你的 safetensors 文件路径
# from your_model_file import YourModelClass
# model = YourModelClass()  # ✅ 替换成你自己的模型定义
model = Pi3()  # ← 这里填入你的模型实例

# === 2️⃣ 加载权重 ===
print(f"\n🚀 Loading weights from: {safetensor_path}")
state_dict = load_file(safetensor_path)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"✅ Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

if missing:
    print("⚠️ Missing keys:", missing[:10])
if unexpected:
    print("⚠️ Unexpected keys:", unexpected[:10])

# === 3️⃣ 查看参数信息 ===
print("\n🔍 Model parameters overview:")
total_params = 0
trainable_params = 0
for name, param in model.named_parameters():
    numel = param.numel()
    total_params += numel
    if param.requires_grad:
        trainable_params += numel
    print(f"{name:<60} shape={tuple(param.shape)} requires_grad={param.requires_grad} params={numel}")

print(f"\n📊 Total parameters: {total_params:,}")
print(f"🧠 Trainable parameters: {trainable_params:,}")
print(f"🧊 Frozen parameters: {total_params - trainable_params:,}\n")


# === 4️⃣ 冻结层示例 ===

# (1) 冻结全部
# for p in model.parameters():
#     p.requires_grad = False

# (2) 冻结部分层
for name, param in model.named_parameters():
    # 比如冻结 encoder、backbone、或者特定模块
    if name.startswith("encoder") or "backbone" in name:
        param.requires_grad = False

# 查看冻结结果
print("\n🔒 Frozen layers after selection:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print("  ", name)

# === 5️⃣ 打印冻结统计 ===
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\n✅ Trainable: {trainable:,} | Frozen: {frozen:,} | Total: {trainable + frozen:,}")
