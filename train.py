from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn


from dataloader import TagFeatureLoader
from model import TagDiscriminator

#-----------------------------------------------------------------------
# parameter setting
# the path to labeled dataset
# the checkpoint can be found in the same file path
labeled_path = "./data/labeled_tag.pkl"
tag_length = 5
#-----------------------------------------------------------------------

bsz = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(TagFeatureLoader(labeled_path), batch_size=bsz, shuffle=True)
model = TagDiscriminator(tag_len=tag_length)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_acm = 0
acc_acm, loss_acm = 0., 0.
stoa_acc = 0.
print_every = 1000
epoch = 50
model.train()

for e in range(1, epoch):
    
    # iteratively read the training dataset
    for i, (feature, label) in enumerate(train_loader):
        batch_acm += bsz

        feature = feature.to(device)
        label = torch.LongTensor([t[0] for t in label])
        label = label.to(device)
        label_pred = model.forward(feature)
        loss = criterion(label_pred, label)
        # print(loss.item())
        loss_acm += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        ans = model.predict(feature).to(device)
        acc = torch.eq(ans, label).sum().item()
        acc_acm += acc

        if batch_acm%print_every == 0:
            print('epoch %d, batch_acm %d, loss %.3f, acc %.3f'\
                %(e, batch_acm, loss_acm/print_every, acc_acm/print_every))
            if acc_acm/print_every > stoa_acc:
                stoa_acc = acc_acm/print_every
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_batch_%dacc_%.3f'%('./data', e, batch_acm, stoa_acc))
            loss_acm, acc_acm = 0., 0.