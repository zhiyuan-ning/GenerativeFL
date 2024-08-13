import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

from record import RecordList
from utils.logger import error, info


base_path = './selection_record'


class Evaluator(object):
    def __init__(self, task_type=None, dataset=None):
        self.records = RecordList()

    def __len__(self):    
        return len(self.records)
  
    
    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

    
    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/choice.pt')
        torch.save(performances, f'{base_path}/history/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/history'):
            os.mkdir(f'{base_path}/history')

 
    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:    
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)    
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):    
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)



    def report_performance(self, choice, performances, store=True, rp=True, flag=''):   
        if store:   
            self._store_history(choice, performances)





