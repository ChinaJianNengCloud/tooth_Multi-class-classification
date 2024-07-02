import timeit
from datetime import datetime
import socket
import glob
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import VideoDataset
from CR_C3D import ResNet50, get_1x_lr_params, get_10x_lr_params
from torch.utils.tensorboard import SummaryWriter
import random
import warnings
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torchmetrics.functional.classification import multilabel_f1_score, multilabel_precision, multilabel_recall, \
    multilabel_accuracy, multilabel_auroc
import requests
import cv2
import torch.nn.functional as F
# token = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjIzOTQ3OSwidXVpZCI6ImEzMTU4MzhmLTIxOTgtNGExYy05ODEzLWNiNWU0NWY2ZTczMSIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.x9mp9ZO6hfuxz1pIAvRiYRZ41GA-DLbpHeCj-hLz8YwNTv601SZIK7HN2fZpTXNLpC3V8MmHND9rGWE-fnPXSA"
#
# headers = {"Authorization": token}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化分布式环境
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
# nccl

def cleanup():
    dist.destroy_process_group()


def calculate_metrics(y_true, y_pred, average_model):
    # 计算精确率、召回率和F1分数
    precision = precision_score(y_true, y_pred, average=average_model)
    recall = recall_score(y_true, y_pred, average=average_model)
    f1 = f1_score(y_true, y_pred, average=average_model)

    return precision, recall, f1


def compute_metrics(num_class, pred, target,threshold):
    # 计算每个标签的指标
    f1_scores = multilabel_f1_score(pred, target, num_labels=num_class, average=None, threshold=threshold)
    precisions = multilabel_precision(pred, target, num_labels=num_class, average=None, threshold=threshold)
    recalls = multilabel_recall(pred, target, num_labels=num_class, average=None, threshold=threshold)
    aurocs = multilabel_auroc(pred, target, num_labels=num_class, average=None)
    accuracies = multilabel_accuracy(pred, target, num_labels=num_class, average=None, threshold=threshold)
    # 创建字典来存储每个标签的指标
    metrics = dict()
    for i in range(num_class):
        metrics[f'f1_{i}'] = f1_scores[i]
        metrics[f'precision_{i}'] = precisions[i]
        metrics[f'recall_{i}'] = recalls[i]
        metrics[f'auroc_{i}'] = aurocs[i]
        metrics[f'accuracy_{i}'] = accuracies[i]
    return metrics

# def compute_metrics(num_classes, pred, target):
#     # 确保target为CPU张量，pred已经在softmax之后
#     preds = pred.argmax(dim=1).cpu().numpy()
#     target_np = target.cpu().numpy()
#
#     # 将target转换为独热编码形式，以便计算AUROC
#     target_one_hot = np.eye(num_classes)[target_np]
#
#     # 计算指标
#     f1_scores = f1_score(target_np, preds, average=None, labels=range(num_classes))
#     precisions = precision_score(target_np, preds, average=None, labels=range(num_classes))
#     recalls = recall_score(target_np, preds, average=None, labels=range(num_classes))
#     accuracy = accuracy_score(target_np, preds)
#
#     # 计算每个类别的AUROC
#     try:
#         aurocs = roc_auc_score(target_one_hot, pred.cpu().detach().numpy(), multi_class="ovr", average=None)
#         # 计算宏平均AUROC
#         macro_auroc = np.mean(aurocs)
#     except ValueError as e:
#         print(f"AUROC计算错误: {e}")
#         aurocs = [np.nan] * num_classes
#         macro_auroc = np.nan
#
#     # 存储指标
#     metrics = {f'f1_{i}': f1_scores[i] for i in range(num_classes)}
#     metrics.update({f'precision_{i}': precisions[i] for i in range(num_classes)})
#     metrics.update({f'recall_{i}': recalls[i] for i in range(num_classes)})
#     metrics.update({f'auroc_{i}': aurocs[i] for i in range(num_classes)})
#     metrics['accuracy'] = accuracy
#     metrics['macro_auroc'] = macro_auroc
#
#     return metrics

weights_group_1 = [1.0, 1.0, 1.0, 1.0 ,1.0, 1.0,1.0]

class_weights_1 = torch.tensor(weights_group_1)

warnings.filterwarnings("ignore")

'''超参数的设定'''
nEpochs = 300  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
'''resume_epoch 表示从什么时候开始训练，默认是0表示从头开始训练'''
average_model = 'samples'
show_interval = 60
batch_size = 64  # 2卡 8

useTest = True  # See evolution of the test set when training
nTestInterval = 100  # Run on test set every nTestInterval epochs
snapshot = 100  # Store a model every snapshot epochs   训练多少次保存一个模型
lr = 1e-4  # Learning rate

dataset = 'oral'
# 标签数量
num_classes = 7
threshold = 0.5
'''这块是打印出名字的路径'''
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
print(save_dir_root, "||", exp_name)

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'ResNet50'  # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset


def update_logging(writer, epoch, loss, acc, auc_scores, metrics, phase):
    writer.add_scalar(f'data/{phase}_loss_epoch', loss, epoch)
    writer.add_scalar(f'data/{phase}_acc_epoch', acc, epoch)

    auc_scores = [np.nan if auc is None else auc for auc in auc_scores]
    writer.add_scalar(f'data/{phase}_mean_AUC', np.nanmean(auc_scores), epoch)
    num_labels = len(metrics) // 5
    avg_f1, avg_precision, avg_recall, avg_auroc, avg_accuracy = 0, 0, 0, 0, 0
    # 记录每个类别的 F1 分数、精确率和召回率，并计算平均值
    for label in range(num_labels):
        f1 = metrics[f'f1_{label}']
        precision = metrics[f'precision_{label}']
        recall = metrics[f'recall_{label}']
        auroc = metrics[f'auroc_{label}']
        accuracy = metrics[f'accuracy_{label}']

        writer.add_scalar(f'data/{phase}_F1_Label_{label}', f1, epoch)
        writer.add_scalar(f'data/{phase}_Precision_Label_{label}', precision, epoch)
        writer.add_scalar(f'data/{phase}_Recall_Label_{label}', recall, epoch)
        writer.add_scalar(f'data/{phase}_Auroc_{label}', auroc, epoch)
        writer.add_scalar(f'data/{phase}_Accuracy_{label}', accuracy, epoch)

        avg_f1 += f1
        avg_precision += precision
        avg_recall += recall
        avg_auroc += auroc
        avg_accuracy += accuracy
        # 计算平均值
    avg_f1 /= num_labels
    avg_precision /= num_labels
    avg_recall /= num_labels
    avg_auroc /= num_labels
    avg_accuracy /= num_labels

    # 记录平均值
    writer.add_scalar(f'data/{phase}_avg_F1', avg_f1, epoch)
    writer.add_scalar(f'data/{phase}_avg_Precision', avg_precision, epoch)
    writer.add_scalar(f'data/{phase}_avg_Recall', avg_recall, epoch)
    writer.add_scalar(f'data/{phase}_avg_Auroc', avg_auroc, epoch)
    writer.add_scalar(f'data/{phase}_avg_Accuracy', avg_accuracy, epoch)


# Function to save the model
def save_model(model, optimizer, epoch, save_dir, saveName):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(
        os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))


def train_model(rank, world_size, dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    setup(rank, world_size)
    torch.cuda.set_device(rank)  # 设置当前使用的 GPU
    device = torch.device(f"cuda:{rank}")
    model = ResNet50(num_classes=num_classes, pretrained=False)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_params = [{'params': get_1x_lr_params(model), 'lr': lr},
                    {'params': get_10x_lr_params(model), 'lr': lr * 10}]

    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_1)
    criterion=nn.CrossEntropyLoss()
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 5e-4

    # 创建Adam优化器
    optimizer = optim.Adam(train_params, lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    train_dataset = VideoDataset(dataset=dataset, split='train', clip_len=128)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=0)

    val_dataset = VideoDataset(dataset=dataset, split='val', clip_len=128)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=0)
    ''' 不要把时间片段、窗口维度和片段维度(clip_len)和 batch_size 搞混'''

    zcx_val_dataset = VideoDataset(dataset=dataset, split='zcx-val', clip_len=128)
    zcx_val_sampler = DistributedSampler(zcx_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    zcx_val_dataloader = DataLoader(zcx_val_dataset, batch_size=batch_size, shuffle=False, sampler=zcx_val_sampler,
                                    num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader, 'zcx-val': zcx_val_dataloader}

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val', 'zcx-val']:
            if phase == 'train':
                train_sampler.set_epoch(epoch)
            else:
                val_sampler.set_epoch(epoch)
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on the phase
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            j = 0
            it_count = 0
            run_loss = 0
            num_samples = 0
            all_preds = []
            all_labels = []
            pred_list = []
            sige_preds = []
            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = (Variable(inputs, requires_grad=False).to(device)).float()
                labels = Variable(labels).to(device)

                # Assuming inputs is a batch of images in NHWC format
                # img = inputs[0].cpu().numpy()  # Move to CPU before converting to NumPy
                # Resize image
                # img_resized = cv2.resize(img, (512, 256))
                # img_resized = np.transpose(img_resized, (2, 0, 1))

                optimizer.zero_grad()

                if phase == 'train':
                    it_count += 1
                    outputs = model(inputs)
                else:
                    j += 1
                    with torch.no_grad():
                        outputs = model(inputs)
                labels_only=torch.argmax(labels,dim=1)

                loss = criterion(outputs, labels_only)
                # 但是用于性能评估的probs和preds仍然需要sigmoid激活
                probs = torch.sigmoid(outputs)

                preds = (probs > threshold).float()
                # probs = F.softmax(outputs, dim=1)
                # _, preds = torch.max(outputs, dim=1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
                run_loss += loss.item()
                probabilities=torch.sigmoid(outputs)



                # probabilities = F.softmax(outputs,dim=1).detach().cpu().numpy()

                # Update the running corrects
                # running_corrects += ((preds == labels).sum(dim=1) == labels.size(1)).sum().item()
                preds_only=torch.argmax(preds,dim=1)
                running_corrects += torch.sum(preds_only==labels_only).item()
                # Storing probabilities and labels
                # all_preds.extend(probabilities)
                # all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                sige_preds.append((probabilities>threshold).float().cpu().numpy())
                pred_list.append(preds)
                if phase == 'train' and it_count % show_interval == 0:
                    print("%d, loss: %.3e, acc: %.3e" % (
                        it_count, run_loss / it_count, running_corrects / (it_count * batch_size)))

            all_preds_tensor = torch.tensor(all_preds)
            all_labels_tensor = torch.tensor(all_labels)
            # all_auc_preds=np.stack(all_preds,axis=0)
            # all_auc_label=np.stack(all_auc_label,axis=0)
            metrics = compute_metrics(num_classes, all_preds_tensor, all_labels_tensor,threshold)

            auc_scores = []
            for i in range(num_classes):
                labels_i = np.array(all_labels)[:, i]
                preds_i = np.array(all_preds)[:, i]
                if len(np.unique(labels_i)) > i:
                    auc_i = roc_auc_score(labels_i, preds_i)
                    auc_scores.append(auc_i)
                else:
                    auc_scores.append(np.nan)  # 对于只有一个类的标签，设置为 None

            average_loss = running_loss / num_samples
            epoch_loss = average_loss
            epoch_acc = running_corrects / num_samples

            if phase == 'train':
                update_logging(writer, epoch, epoch_loss, epoch_acc, auc_scores, metrics, phase)
                print(
                    f"[Train:{phase}] Epoch:{epoch + 1}/{nEpochs}||Loss:{epoch_loss}||Acc:{epoch_acc}||AUC:{auc_scores}")
                for label in range(len(metrics) // 5):
                    print(
                        f"Label {label} - F1: {metrics[f'f1_{label}']}, Precision: {metrics[f'precision_{label}']}, Recall: {metrics[f'recall_{label}']}, Auroc:{metrics[f'auroc_{label}']},Accuracy:{metrics[f'accuracy_{label}']}")
            if phase == 'val':
                update_logging(writer, epoch, epoch_loss, epoch_acc, auc_scores, metrics, phase)
                print(
                    f"[Val:{phase}] Epoch:{epoch + 1}/{nEpochs}||Loss:{epoch_loss}||Acc:{epoch_acc}||AUC:{auc_scores}")
                for label in range(len(metrics) // 5):
                    print(
                        f"Label {label} - F1: {metrics[f'f1_{label}']}, Precision: {metrics[f'precision_{label}']}, Recall: {metrics[f'recall_{label}']}, Auroc:{metrics[f'auroc_{label}']},Accuracy:{metrics[f'accuracy_{label}']}")
            if phase == 'zcx-val':
                update_logging(writer, epoch, epoch_loss, epoch_acc, auc_scores, metrics, phase)
                print(
                    f"[zcx-Val:{phase}] Epoch:{epoch + 1}/{nEpochs}||Loss:{epoch_loss}||Acc:{epoch_acc}||AUC:{auc_scores}")
                for label in range(len(metrics) // 5):
                    print(
                        f"Label {label} - F1: {metrics[f'f1_{label}']}, Precision: {metrics[f'precision_{label}']}, Recall: {metrics[f'recall_{label}']}, Auroc:{metrics[f'auroc_{label}']},Accuracy:{metrics[f'accuracy_{label}']}")

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if rank == 0 and epoch % save_epoch == (save_epoch - 1):
            save_model(model, optimizer, epoch, save_dir, saveName)
    writer.close()


def main():
    world_size = 1
    mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

