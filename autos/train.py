import argparse
import os
import sys

import pandas
sys.path.append("./IJCAI-AutoS")
import pickle
import random
import sys
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from autos.model import AUTOS
from device_env import Evaluator, base_path
from autos.train_utils import AvgrageMeter, pairwise_accuracy, hamming_distance, FSDataset
from record import SelectionRecord
from utils.logger import info, error
from plato.config import Config
from torchstat import stat

# parser = argparse.ArgumentParser()
# parser.add_argument('--random_seed', type=int, default=1)
# parser.add_argument('--new_gen', type=int, default=100)
# parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')
# parser.add_argument('--gpu', type=int, default=0, help='used gpu')
# parser.add_argument('--top_k', type=int, default=100)
# parser.add_argument('--gen_num', type=int, default=25)
# parser.add_argument('--encoder_layers', type=int, default=1)
# parser.add_argument('--encoder_hidden_size', type=int, default=64)
# parser.add_argument('--encoder_emb_size', type=int, default=32)
# parser.add_argument('--mlp_layers', type=int, default=2)
# parser.add_argument('--mlp_hidden_size', type=int, default=200)
# parser.add_argument('--decoder_layers', type=int, default=1)
# parser.add_argument('--decoder_hidden_size', type=int, default=64)
# parser.add_argument('--encoder_dropout', type=float, default=0)
# parser.add_argument('--mlp_dropout', type=float, default=0)
# parser.add_argument('--decoder_dropout', type=float, default=0)
# parser.add_argument('--l2_reg', type=float, default=0.0)
# parser.add_argument('--max_step_size', type=int, default=100)
# parser.add_argument('--trade_off', type=float, default=0.8)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--batch_size', type=int, default=1024)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--optimizer', type=str, default='adam')
# parser.add_argument('--grad_bound', type=float, default=5.0)
# args = parser.parse_args()




def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def autos_train(train_queue, model: AUTOS, optimizer):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']

        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch = model.forward(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)) # ce loss
        loss = Config().server.autos.trade_off * loss_1 + (1 - Config().server.autos.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config().server.autos.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


def autos_valid(queue, model: AUTOS):
    pa = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch = model.forward(encoder_input)
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                             predict_value.data.squeeze().tolist())
            hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return mse.avg, pa.avg, hs.avg


def choice_to_onehot(choice: List[int]):
    size = len(choice)
    onehot = torch.zeros(size + 1)
    onehot[torch.tensor(choice)] = 1
    return onehot[:-1]
    # if choice.dim() == 1:
    #     selected = torch.zeros_like(choice)
    #     selected[choice] = 1
    #     return selected[1:-1]
    # else:
    #     onehot = torch.empty_like(choice)
    #     for i in range(choice.shape[0]):
    #         onehot[i] = choice_to_onehot(choice[i])
    #     return onehot


def autos_infer(queue, model, step, direction='+'):
    new_gen_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda(model.gpu)
        model.zero_grad()
        new_gen = model.generate_new_device(encoder_input, predict_lambda=step, direction=direction)
        new_gen_list.extend(new_gen.data.squeeze().tolist())
    return new_gen_list


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def autos_selection(device_pool_num, model_state_dict):
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in Config().server.autos.gpu)
    random.seed(Config().server.autos.random_seed)
    np.random.seed(Config().server.autos.random_seed)
    torch.manual_seed(Config().server.autos.random_seed)
    torch.cuda.manual_seed(Config().server.autos.random_seed)
    torch.cuda.manual_seed_all(Config().server.autos.random_seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(Config().server.autos.gpu)
    # info(f"Args = {args}")

    with open(f'{base_path}/history/device_env.pkl', 'rb') as f:
        de: Evaluator = pickle.load(f)
    
   
    model = AUTOS(Config)
    # model_path = "./autos/rnn.pth"
    # pretrained = None
    # if model_state_dict is None:
    #     pass
    # else:
    #     pretrained = torch.load(model_path)
    #     model.load_state_dict(pretrained, strict=True)

    info(f"param size = {count_parameters_in_MB(model)}MB")
    model = model.cuda(device)

    choice, labels = de.get_record(25, eos=device_pool_num)

    valid_choice, valid_labels = de.get_record(0, eos=device_pool_num)

    info('Training Encoder-Predictor-Decoder')

    min_val = min(labels)
    max_val = max(labels)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]

    train_dataset = FSDataset(choice, train_encoder_target, train=True, sos_id=device_pool_num, eos_id=device_pool_num)
    valid_dataset = FSDataset(valid_choice, valid_encoder_target, train=False, sos_id=device_pool_num, eos_id=device_pool_num)

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config().server.autos.batch_size, shuffle=True, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=Config().server.autos.lr, weight_decay=Config().server.autos.l2_reg)
    for epoch in range(1, Config().server.autos.epochs + 1):
        loss, mse, ce = autos_train(train_queue, model, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, loss, mse, ce))
        if epoch % 100 == 0 or epoch == 1:
            mse, pa, hs = autos_valid(train_queue, model)
            info("Evaluation on train data")
            info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
                                                                                                   hs))
            mse, pa, hs = autos_valid(valid_queue, model)
            info("Evaluation on valid data")
            info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
                                                                                                   hs))

    top_selection, top_performance = select_top_k(valid_choice, valid_labels, Config().server.autos.top_k)

    # torch.save(model.state_dict(), model_path)


    infer_dataset = FSDataset(top_selection, top_performance, False, sos_id=device_pool_num, eos_id=device_pool_num)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)
    new_selection = []
    new_choice = []
    predict_step_size = 0
    while len(new_selection) < Config().server.autos.new_gen:
        predict_step_size += 1
        info('Generate new architectures with step size {:d}'.format(predict_step_size))
        new_record = autos_infer(infer_queue, model, direction='+', step=predict_step_size)
        for choice in new_record:
            onehot_choice = choice_to_onehot(choice)
            if onehot_choice.sum() <= 0:
                error('insufficient selection')
                continue
            record = SelectionRecord(onehot_choice.numpy(), -1)
            if record not in de.records.r_list and record not in new_selection:
                new_selection.append(record)
                new_choice.append(onehot_choice)
            if len(new_selection) >= Config().server.autos.new_gen:
                break
        info(f'{len(new_selection)} new choice generated now', )
        if predict_step_size > Config().server.autos.max_step_size:
            break
    info(f'build {len(new_selection)} new choice !!!')

    new_choice_pt = torch.stack(new_choice)
    if Config().server.autos.gen_num == 0:
        choice_path = f'{base_path}/history/generated_choice.pt'
    else:
        choice_path = f'{base_path}/history/generated_choice.pt'
    torch.save(new_choice_pt, choice_path)
    info(f'save generated choice to {choice_path}')
    
    return new_selection, model_state_dict

    
    # torch.save(model.state_dict(), f'{base_path}/history/model_dict')

    # best_selection_test = None
    # best_optimal_test = -1000
    # for s in new_selection:
    #     test_data = de.generate_data(s.operation, 'test')
    #     test_result = de.get_performance(test_data)
    #     if test_result > best_optimal_test:
    #         best_selection_test = s.operation
    #         best_optimal_test = test_result
    #         info(f'found best on test : {best_optimal_test}')

    # info(f'found test generation in our method! the choice is {best_selection_test}')
    


# if __name__ == '__main__':
#     device_pool_num =200
#     autos_selection(200)

