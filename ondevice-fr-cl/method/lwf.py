from copy import deepcopy

import torch
import torch.nn.functional as F

from torch.optim import SGD
from torchvision.transforms import transforms

from models import ContinualModel


__all__ = ['LwF']

def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class LwF(ContinualModel):
    NAME = 'lwf'

    def __init__(self, net, loss, args, transform):
        super(LwF, self).__init__(net, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.current_task = 0
        self.cpt = args.N_CLASSES_PER_TASK
        nc = args._DEFAULT_N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)
        self.totensor = transforms.ToTensor()

    def begin_task(self, loader):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.classifier.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(loader):
                    inputs, labels, task, not_aug_inputs = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net(inputs, outputs='features')
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    outputs = self.net.classifier(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(0, loader.dataset.dataset.data.shape[0], self.args.batch_size):
                    inputs = torch.stack([self.totensor(loader.dataset.dataset.__getitem__(j)[0])
                                          for j in range(i, min(i + self.args.batch_size,
                                                         len(loader.dataset.dataset)))])
                    log = self.net(inputs.to(self.device)).cpu()
                    logits.append(log)
            setattr(loader.dataset.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        self.optim.zero_grad()
        outputs = self.net(inputs, outputs='logits')

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[:, mask], labels)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                                                      smooth(self.soft(outputs[:, mask]), 2, 1))

        loss.backward()
        self.optim.step()

        return loss.item()