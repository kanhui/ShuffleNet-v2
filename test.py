# -*- coding: utf-8 -*-

"""
@Date: 2018/12/25

@Author: dreamhome

@Summary:  
"""
import torch
import numpy as np
from torch.autograd import Variable

a = np.array([[1.0, 2.0, 3.0],
              [2.0, 3.0, 4.0],
              [3.0, 4.0, 5.0],
              [4.0, 5.0, 6.0]])
a = torch.from_numpy(a)
print(a)
print(a.data)
print(a.detach())
# labels = Variable(torch.from_numpy(np.array([0, 0, 0, 0]))).long()
# criterion = torch.nn.CrossEntropyLoss()
# print(a.data.max(1)[1])
# print(criterion(a, labels).data)