# import dependencies
import pickle

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import visdom

import evaluate


# define function
def train(model, train_loader, train_size, val_loader, val_size, criterion,
          optimizer, scheduler, epochs, device, model_path, checkpoint,
          hist_path, resume, visdom, environment, pbar_file):
    '''train model'''
    # setup loss and accuracy visualization
    if visdom:
        viz = visdom.Visdom()
        loss_plot, acc_plot = None, None
    else:
        train_loss, val_loss, train_acc, val_acc = {}, {}, {}, {}

    # load checkpoints, if training to be resumed
    if resume:
        checkpoint_dict = torch.load(checkpoint)

        best_val_loss = checkpoint_dict['best_val_loss']
        no_improvement = checkpoint_dict['no_improvement']
        start_epoch = checkpoint_dict['epoch']

        if not visdom:
            train_loss = checkpoint_dict['train_loss']
            val_loss = checkpoint_dict['val_loss']
            train_acc = checkpoint_dict['train_acc']
            val_acc = checkpoint_dict['val_acc']
    else:
        best_val_loss = float('inf')
        no_improvement = 0
        start_epoch = 1

    # train
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        # save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'no_improvement': no_improvement,
                'train_loss': train_loss if not visdom else None,
                'val_loss': val_loss if not visdom else None,
                'train_acc': train_acc if not visdom else None,
                'val_acc': val_acc if not visdom else None
            }, checkpoint)

        # setup progress bar
        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm.tqdm(desc=desc.format(0),
                         total=len(train_loader),
                         leave=False,
                         file=pbar_file,
                         initial=0)

        tqdm.tqdm.write('epoch {} of {}'.format(epoch, epochs), file=pbar_file)

        # iterate
        batch_idx = 0
        running_loss = 0
        correct = 0
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimize and save stats
            optimizer.zero_grad()

            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels).item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            # evaluate model performance and invoke learning rate scheduler
            if batch_idx == (len(train_loader) - 1):
                avg_val_loss, val_correct = evaluate.evaluate(
                    model, val_loader, val_size, criterion, device, False,
                    pbar_file)
                scheduler.step(metrics=avg_val_loss)

            # update progress bar
            pbar.desc = desc.format(loss.item())
            pbar.update()

        # print epoch end losses and accuracies
        tqdm.tqdm.write('training loss: {:.4f}, val loss: {:.4f}'.format(
            running_loss / (batch_idx + 1), avg_val_loss, file=pbar_file))
        tqdm.tqdm.write('training acc: {:.2f}%, val acc: {:.2f}%\n'.format(
            (correct * 100) / train_size, (val_correct * 100) / val_size,
            file=pbar_file))

        # close progress bar
        pbar.close()

        # plot loss history
        if visdom:
            if not loss_plot:
                loss_plot = viz.line(X=np.array([epoch]),
                                     Y=np.array(
                                         [running_loss / (batch_idx + 1)]),
                                     env=environment,
                                     opts=dict(legend=['train', 'val'],
                                               title='loss hist',
                                               xlabel='epochs',
                                               ylabel='loss'))
            else:
                viz.line(X=np.array([epoch]),
                         Y=np.array([running_loss / (batch_idx + 1)]),
                         env=environment,
                         win=loss_plot,
                         name='train',
                         update='append')

            if not acc_plot:
                acc_plot = viz.line(X=np.array([epoch]),
                                    Y=np.array([(correct * 100) / train_size]),
                                    env=environment,
                                    opts=dict(legend=['train', 'val'],
                                              title='acc hist',
                                              xlabel='epochs',
                                              ylabel='acc'))
            else:
                viz.line(X=np.array([epoch]),
                         Y=np.array([(correct * 100) / train_size]),
                         env=environment,
                         win=acc_plot,
                         name='train',
                         update='append')
        else:
            train_loss[epoch] = running_loss / (batch_idx + 1)
            val_loss[epoch] = avg_val_loss
            train_acc[epoch] = (correct * 100) / train_size
            val_acc[epoch] = (val_correct * 100) / val_size

        # save model and apply early stopping
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), model_path)
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == 5:
            print('applying early stopping')
            break

    # save training history
    if not visdom:
        hist = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        with open(hist_path, 'wb') as hist_file:
            pickle.dump(hist, hist_file)

    # visualize losses and accuracies
    if not visdom:
        for subplot in ['loss', 'acc']:
            if subplot == 'loss':
                plt.subplot(1, 2, 1)
            else:
                plt.subplot(1, 2, 2)

            plt.title(subplot)
            plt.xlabel = 'epochs'
            plt.ylabel = 'loss' if subplot == 'loss' else 'acc'

            train_plot, = plt.plot(train_loss.values() if subplot == 'loss'
                                   else train_acc.values(),
                                   label='train')
            val_plot, = plt.plot(
                val_loss.values() if subplot == 'loss' else val_acc.values(),
                label='val')
            plt.legend(handles=[train_plot, val_plot], loc='best')

        plt.tight_layout()
        plt.show()
