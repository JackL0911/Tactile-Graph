import torch
from torch.utils.data import random_split
from contrastive_learning_dataset import ContrastiveLearningDataset
import os
from tqdm import tqdm
import logging
from utils import accuracy, save_checkpoint
from model import MultiModalMoCo

config = {
    "train_dataset_name": 'tag_train',
    "data_folder": "dataset/dataset",
    "model_name": "TAG",
    "num_channels": 3,
    "epochs": 100,
    "log_every_n_epochs": 10,
    "batch_size": 32,
    "num_workers": 16,
    "momentum": 0.99,
    "temperature": 0.07,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "nn_model": 'resnet18',
    "intra_dim": 128,
    "inter_dim": 128,
    "weight_inter_tv": 1,
    "weight_inter_vt": 1,
    "weight_intra_vision": 1,
    "weight_intra_tactile": 1,
    "pretrained_encoder": True,
}

# 设置日志记录
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

dataset = ContrastiveLearningDataset(root_folder=config['data_folder'])
train_dataset = dataset.get_dataset(config['train_dataset_name'], 2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                           num_workers=config['num_workers'], drop_last=False, pin_memory=True)

# 加载模型
model = MultiModalMoCo(n_channels=config['num_channels'], m=config['momentum'], T=config['temperature'],
                       intra_dim=config['intra_dim'], inter_dim=config['inter_dim'], nn_model=config['nn_model'],
                       weight_inter_tv=config['weight_inter_tv'], weight_inter_vt=config['weight_inter_vt'],
                       weight_intra_vision=config['weight_intra_vision'], weight_intra_tactile=config['weight_intra_tactile'],
                       pretrained_encoder=config['pretrained_encoder'])

# 定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training with gpu: {device}.")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
criterion = torch.nn.CrossEntropyLoss().to(device)

best_acc = 0
for epoch in range(config['epochs']):
    loss_epoch, vis_loss_intra_epoch, tac_loss_intra_epoch, vis_tac_inter_epoch, tac_vis_inter_epoch = 0, 0, 0, 0, 0
    pbar = tqdm(train_loader)
    for idx, values in enumerate(pbar):
        x_vision_q_list, x_vision_k_list, x_tactile_q_list, x_tactile_k_list, label = values
        model.train()

        x_vision_q = torch.stack(x_vision_q_list, dim=0)
        x_vision_k = torch.stack(x_vision_k_list, dim=0)
        x_tactile_q = torch.stack(x_tactile_q_list, dim=0)
        x_tactile_k = torch.stack(x_tactile_k_list, dim=0)

        x_vision_q = x_vision_q.transpose(0, 1)  # Shape: (32, 8, 3, 224, 224)
        x_vision_k = x_vision_k.transpose(0, 1)
        x_tactile_q = x_tactile_q.transpose(0, 1)
        x_tactile_k = x_tactile_k.transpose(0, 1)

        model.train()
        x_vision_q = x_vision_q.to(device, non_blocking=True)
        x_vision_k = x_vision_k.to(device, non_blocking=True)
        x_tactile_q = x_tactile_q.to(device, non_blocking=True)
        x_tactile_k = x_tactile_k.to(device, non_blocking=True)

        loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter, logits, labels = model(x_vision_q, x_vision_k, x_tactile_q, x_tactile_k)
        loss_epoch += loss.item()
        vis_loss_intra_epoch += vis_loss_intra.item()
        tac_loss_intra_epoch += tac_loss_intra.item()
        vis_tac_inter_epoch += vis_tac_inter.item()
        tac_vis_inter_epoch += tac_vis_inter.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}:\tTrain loss: {loss_epoch / (idx + 1):.2f}")

    if epoch % config['log_every_n_epochs'] == 0: # topk 用来衡量在前 k 个最有可能的预测中，正确答案是否包含在其中。

        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        print("\n")
        logger.info(f"Epoch {epoch} - Loss: {loss_epoch / len(train_loader):.4f} - Top1 Accuracy: {top1[0]:.2f}% - Top5 Accuracy: {top5[0]:.2f}%")
        if top1[0] > best_acc:
            best_acc = top1[0]
            save_checkpoint({
                'epoch': epoch,
                'arch': 'resnet18',
                'model_state_dict': model.state_dict(),  # 保存整个模型的参数
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(log_dir, f'model_best_{epoch}.pth'))

    if epoch >= 10:
        scheduler.step()

logger.info("Training has finished.")
checkpoint_name = f'checkpoint_{config["epochs"]:04d}.pth.tar'
save_checkpoint({
    'epoch': config['epochs'],
    'arch': config['nn_model'],
    'model_state_dict': model.state_dict(),  # 保存整个模型的参数
    'optimizer': optimizer.state_dict(),
}, filename=os.path.join(log_dir, checkpoint_name))
logger.info(f"Model checkpoint and metadata have been saved at {log_dir}.")
