#这里是一个epoch发生的训练
import time
import logging
from .val_function import validate
import psutil
import os

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))

def train_epoch(args, train_loader, model, criterion, optimizer):
    model.train()
    loss = 1
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()  #梯度归零
        outputs = model(inputs)  #模型输出（分类情况）
        loss = criterion(outputs, targets)  #根据真实类别和模型划分的类别情况计算损失
        loss.backward()  #反向传播计算得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行一步参数更新
    # print(f"loss={loss.item():.4f}")

#我们还需要一个总的训练函数，一次调用就代表我们实验重复一次
def train(args, iter, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_num):
    start = time.time()
    best_val_OA = 0.0   #测试集最好的总体分类精度
    best_model = None   #记录性能最好的模型的权重

    # 记录初始内存
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / 1024 ** 2  # MB

    for epoch in range(epoch_num):
        # Choice Model Training
        print('-------Iter:{}....train weights for epoch:{}-------'.format(iter, epoch + 1))
        train_epoch(args, train_loader, model, criterion, optimizer)
        #更新学习率计划
        scheduler.step()
        # Choice Model Validation 验证模型
        val_OA = validate(model, val_loader)  #验证集的总体分类精度
        print('val weights for epoch:{},val_OA={}'.format(epoch + 1, val_OA))

        # 打印当前内存
        current_mem = process.memory_info().rss / 1024 ** 2
        print(f"[Memory] Current RAM Usage: {current_mem:.2f} MB | Delta: {current_mem - initial_mem:.2f} MB")


        # Save Best Model Weights  保存最佳模型权重参数
        if best_val_OA <= val_OA:
            best_val_OA = val_OA
            best_model = model
            print('Best val_OA by now,update the best model')

    # 输出最终内存
    peak_mem = process.memory_info().rss / 1024 ** 2
    print(f"\n[Final Memory] Initial: {initial_mem:.2f} MB | Peak: {peak_mem:.2f} MB | Total Used: {peak_mem - initial_mem:.2f} MB")

    print('*****Iter:{}....This iteration has ended, best_val_OA:{}, , return the best model*****'.format(iter, best_val_OA))
    # Record Time
    time_record(start)
    #这里只需要返回我们训练的最好权重即可
    return best_model