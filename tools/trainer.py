import numpy as np
import torch

def train_epoch(model,dataloader,loss,optimizer,device):
    model.train()
    acc = []
    lss_history = []
    for _ , (data,labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(data)
        lss = loss(pred,labels)

        lss.backward()
        optimizer.step()


        # acc calculations
        lss_history.append(lss.item())
        acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())


    return np.mean(lss_history) ,np.mean(acc)

def validate_epoch(model,dataloader,loss,device):
    model.eval()
    acc = []
    lss_history = []
    with torch.no_grad():
        for i , (data,labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            lss = loss(pred,labels)
            lss_history.append(lss.item())
            acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())
    return np.mean(lss_history) ,np.mean(acc)



def tune_model(num_epochs,model,train_dataloader_,test_dataloader_,\
               loss,optimizer,device,scheduler=None,earlystopping=None) :
    '''
    NOTE that the scheduler here takes the update after every epoch not step
    '''

    hist = {'train_loss': [],
            'train_acc':[],
            'test_loss': [],
            'test_acc':[]}

    last_lr = optimizer.state_dict()['param_groups'][0]['lr']
    f= 0
    for e in range(num_epochs):
        lss,acc= train_epoch(model,train_dataloader_,loss,optimizer,device)
        test_lss,test_acc= validate_epoch(model,test_dataloader_,loss,device)


        if (e + 1) % 5==0:
            print(f"For epoch {e:3d} || Training Loss {lss:5.3f} || acc {acc:5.3f}",end='')
            print(f" || Testing Loss {test_lss:5.3f} || Test acc {test_acc:5.3f}")
        hist['train_loss'].append(lss)
        hist['train_acc'].append(acc)
        hist['test_loss'].append(test_lss)
        hist['test_acc'].append(test_acc)



        #
        if earlystopping:
            if earlystopping(model,test_lss): # should terminate
                print('Early Stopping Activated')
                return hist
        # if you have scheduler
        if scheduler:
            scheduler.step(test_lss)
            try:
        # applying manual verbose for the scheduler
                if last_lr != scheduler.get_last_lr()[0]:
                    print(f'scheduler update at Epoch {e+1}')
                    last_lr = scheduler.get_last_lr()[0]
            except:
                f+=1
    return hist
