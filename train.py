from config import *
from utils import *
from model import *
import os

if __name__ == '__main__':
    # Set the batch_size to 50.
    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    dev_dataset = Dataset('dev')
    dev_loader = data.DataLoader(dev_dataset, batch_size=50, shuffle=True)

    # Use TextCNN model
    model = TextCNN().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 50 iterations, evaluate the model performance using a custom evaluate function
            if b % 50 != 0:
                continue

            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)

            with torch.no_grad():
                # After shuffling, obtain one batch of validation data.
                dev_input, dev_mask, dev_target = next(iter(dev_loader))

                dev_input = dev_input.to(DEVICE)
                dev_mask = dev_mask.to(DEVICE)
                dev_target = dev_target.to(DEVICE)
                dev_pred = model(dev_input, dev_mask)
                dev_pred_ = torch.argmax(dev_pred, dim=1)
                dev_report = evaluate(dev_pred_.cpu().data.numpy(), dev_target.cpu().data.numpy(), output_dict=True)

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', report['accuracy'],
                'dev_acc:', dev_report['accuracy'],
            )
        # Save the model at the end of each epoch.
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model, MODEL_DIR + f'{e}.pth')
