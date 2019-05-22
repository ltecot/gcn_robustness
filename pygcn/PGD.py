import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import load_data, accuracy
import time
class PGD():
    def __init__(self,model):
        self.model = model

    def get_loss(self,features,labels, idx_test, TARGETED, multitask):
        output = self.model(features)
        #print(output, label_or_target)
        if multitask:
            acc_test = torch.mean(correct.float())
            criterion = torch.nn.BCEWithLogitsLoss()
            loss_test = criterion(preds, labels)
        else: 
            loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        #print(output[idx_test])
        #print(loss)
        #print(c.size(),modifier.size())
        return loss_test, acc_test

    def pgd_sp(self, features, labels, idx_test, mask, epsilon, eta, TARGETED=False, multitask=False):
        x_in = features
        yi = Variable(labels)
        x_adv = x_in
        diff = Variable(features[idx_test],requires_grad=True)
        for it in range(100):
            x_adv[idx_test] = x_in[idx_test] + diff
            #t1=time.time()
            error, acc = self.get_loss(x_adv,yi, idx_test, TARGETED, multitask)
            #t2=time.time()
            #print("Time prediction span {}".format(t2-t1))
 
            #if (it)%1==0:
            #    print(error.data.item(), acc.data.item()) 
            if acc.data.item()==0.0:
                break
            #x_adv.grad.data.zero_()
            error.backward(retain_graph=True)
            #t3 = time.time()
            #print("Time backprop span {}".format(t3-t2))
            #print(gradient)
            #print(x_adv.grad.size())
            #masked_grad= x_adv.grad*mask
            #masked_grad.sign_()
            if TARGETED:
                #x_adv.data = x_adv.data - eta* epsilon * masked_grad
                diff.data = diff.data - eta* epsilon * diff.grad
            else:
                #x_adv.data = x_adv.data + eta* epsilon * masked_grad
                diff.data = diff.data + eta* epsilon * diff.grad
            diff.data.clamp_(-epsilon,epsilon)
            #x_adv.data=(diff + x_in).clamp_(0, 1)
            x_adv.grad.data.zero_()
        return diff


    def pgd(self, features, labels, idx_test, mask, epsilon, eta, TARGETED=False, multitask=False):
        x_in = features
        yi = Variable(labels)
        x_adv = Variable(features, requires_grad=True)
        for it in range(100):
            error, acc = self.get_loss(x_adv,yi, idx_test, TARGETED, multitask)
            #if (it)%1==0:
            #    print(error.data.item(), acc.data.item()) 
            #x_adv.grad.data.zero_()
            if acc.data.item()==0.0:
                break
            error.backward(retain_graph=True)
            #print(gradient)
            #print(x_adv.grad.size())
            masked_grad= x_adv.grad*mask
            masked_grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* epsilon * masked_grad
            else:
                x_adv.data = x_adv.data + eta* epsilon * masked_grad
            diff = x_adv.data - x_in
            diff.clamp_(-epsilon,epsilon)
            x_adv.data=(diff + x_in).clamp_(0, 1)
            x_adv.grad.data.zero_()
        return x_adv

    def __call__(self, features, labels, idx_test, mask, epsilon, eta=0.1, TARGETED=False, sparse=False, multitask=False):
        if sparse:
            adv = self.pgd_sp(features, labels, idx_test, mask, epsilon, eta, TARGETED, multitask)
        else:    
            adv = self.pgd(features, labels, idx_test, mask, epsilon, eta, TARGETED, multitask)
        return adv  


