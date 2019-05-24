# import
import numpy as np
import sklearn as skl
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import torch
import torch.distributions.kl as kl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm


# model
class CAE_ENC(nn.Module):
    def __init__(self):
        super().__init__()
        # self.enc = nn.Sequential(*list(model.features.children())[:-5])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1000)

    def forward(self, x):
        # x = self.features(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc1(x)
        return x


class CAE_DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(1000, 256 * 6 * 6)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 2, stride=2)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=1)  # might have to remove

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = x.view(128, 256, 6, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.conv5(x))  # might have to remove
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, weights=None, alpha=1.0):
        super().__init__()
        if weights:
            self.weights = weights
        else:
            self.weights = torch.empty(1000, 1000)
            nn.init.xavier_uniform_(self.weights)
        self.alpha = alpha

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum(
            (x.unsqueeze(1) - self.weights)**2, dim=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(
            torch.transpose(q, 1, 2) / torch.sum(q, dim=1), 1, 2)
        return q


def set_weights(module, weights):
    if isinstance(module, ClusteringLayer):
        module.weights = weights


class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CAE_ENC()
        self.dec = CAE_DEC()
        self.clus = ClusteringLayer()

    def forward(self, x):
        h = self.enc(x)
        q = self.clus(h)
        o = self.dec(h)
        return (h, q, o)


def loss(q, p, o, gamma=0.1):
    mse = nn.MSELoss(o)
    kld = gamma * kl.kl_divergence(p, q)
    l = mse + kld
    return l


def target_distribution(q):
    weight = q**2 / torch.sum(q, dim=0)
    return torch.transpose(torch.transpose(q) / torch.sum(weight, dim=1))


# data
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                         inplace=True)
])
dataset1 = datasets.ImageFolder('/beegfs/vag273/ssl_data_96/supervised/train/',
                                transform=transformations)
dataset2 = datasets.ImageFolder('/beegfs/vag273/ssl_data_96/unsupervised/',
                                transform=transformations)
dataset = data.ConcatDataset((dataset1, dataset2))

train_ratio = 0.9
train_set_size = int(train_ratio * len(dataset))
val_set_size = len(dataset) - train_set_size

train_data, val_data = data.random_split(dataset,
                                         (train_set_size, val_set_size))

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=128, shuffle=False)

# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CAE().to(device)
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# pretrain
best_val_loss = float('inf')
tot_epochs = 200  # maybe lower it on one of the runs
print('pretrain')
for epoch in range(tot_epochs):
    model.train()

    print('epoch {} of {}'.format(epoch + 1, tot_epochs))

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm.tqdm(desc=desc.format(0),
                     total=len(train_loader),
                     leave=False,
                     file=None,
                     initial=0)

    running_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img, _ = data
        img = img.to(device)

        optimizer.zero_grad()

        _, _, out = model(img)
        loss = nn.MSELoss(out, img)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        pbar.desc = desc.format(loss.item())
        pbar.update()

    print('loss: {}'.format(running_loss / len(train_loader)))

    model.eval()
    with torch.no_grad():
        val_running_loss = 0
        for val_batch_idx, val_data in enumerate(val_loader):
            val_img, _ = val_data
            val_img = val_img.to(device)

            _, _, val_out = model(val_img)
            val_loss = nn.MSELoss(val_out, val_img)

            val_running_loss += val_loss.item()

        if val_running_loss / len(val_loader) < best_val_loss:
            torch.save(model.state_dict(), 'weights.pth')

        print('val loss: {}'.format(val_running_loss / len(val_loader)))

    pbar.close()

# first cluster
features = None
for batch_idx, data in enumerate(train_loader):
    img, _ = data
    img = img.to(device)

    if not features:
        features = model(img)
    else:
        torch.cat((features, model(img)), 0)

kmeans = cluster.kMeans(n_clusters=1000, n_init=20)
features = features.view(-1)
pred_last = kmeans.fit_predict(features)
q = kmeans.cluster_centers_

# deep cluster
print('deep cklustering')
update_interval = 140  # maybe reduce this for sake of time
maxiter = 20000  # maybe reduce this for sake of time
for ite in range(int(maxiter)):
    model.train()
    if ite % update_interval == 0:
        q = None
        for batch_idx, data in enumerate(train_loader):
            img, _ = data
            img = img.to(device)

            if not features:
                _, q, _ = model(img)
            else:
                _, new_q, _ = model(img)
                torch.cat((q, new_q), 0)
        p = target_distribution(
            q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        pred = q.argmax(1)

        # check stop criterion
        delta_label = np.sum(pred != pred_last).astype(
            np.float32) / pred.shape[0]
        pred_last = np.copy(pred)
        if ite > 0 and delta_label < 0.001:  # 0.001 is the tolerance
            print('delta_label ', delta_label, '< tol ', 0.001)  # tol
            print('Reached tolerance threshold. Stopping training.')
            break

    print('epoch {} of {}'.format(epoch + 1, tot_epochs))

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm.tqdm(desc=desc.format(0),
                     total=len(train_loader),
                     leave=False,
                     file=None,
                     initial=0)

    running_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img, _ = data
        img = img.to(device)

        optimizer.zero_grad()

        _, q, out = model(img)
        loss = loss(q,
                    p[batch_idx * 128:batch_idx * 128 + 128, :],
                    out,
                    gamma=0.1)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        pbar.desc = desc.format(loss.item())
        pbar.update()

    print('loss: {}'.format(running_loss / len(train_loader)))

    model.eval()
    with torch.no_grad():
        val_running_loss = 0
        for val_batch_idx, val_data in enumerate(val_loader):
            val_img, _ = val_data
            val_img = val_img.to(device)

            _, val_q, val_out = model(val_img)
            val_loss = loss(val_q,
                            p[val_batch_idx * 128:val_batch_idx * 128 +
                              128, :],
                            val_out,
                            gamma=0.1)

            val_running_loss += val_loss.item()

        if val_running_loss / len(val_loader) < best_val_loss:
            torch.save(model.state_dict(), 'overall_weights.pth')

        print('val loss: {}'.format(val_running_loss / len(val_loader)))

    pbar.close()
