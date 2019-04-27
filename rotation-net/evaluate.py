# import dependencies
import torch
import tqdm


# define function
def evaluate(model, dataloader, data_size, criterion, device, pbar, pbar_file):
    '''evaluate model performance'''
    model.eval()

    # setup progress bar
    if pbar:
        desc = "ITERATION - loss: {:.2f}"
        pbar_ = tqdm.tqdm(desc=desc.format(0),
                          total=len(dataloader),
                          leave=False,
                          file=pbar_file,
                          initial=0)

    with torch.no_grad():
        running_loss = 0
        correct = 0
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels).item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # update progress bar
            if pbar:
                pbar_.desc = desc.format(loss.item())
                pbar_.update()

        avg_loss = running_loss / (batch_idx + 1)

        # close progress bar and print stats
        if pbar:
            pbar_.close()
            tqdm.tqdm.write('val loss: {:.2f}, val accuracy {:.2f}%\n'.format(
                running_loss / (batch_idx + 1), (correct * 100) / data_size),
                            file=pbar_file)

    return avg_loss, correct
