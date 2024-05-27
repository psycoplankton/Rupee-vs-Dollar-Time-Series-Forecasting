from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
# TensorBoard setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/experiment_{timestamp}')
loss_train, loss_val = [], []
def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_train.append(avg_loss)
    print(f'Epoch [{epoch_index + 1}], Training loss: {avg_loss:.4f}')
    return avg_loss

def validate():
    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            vloss = loss_fn(outputs, labels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(test_loader)
    loss_val.append(avg_vloss)
    print(f'Epoch [{epoch_index + 1}], Validation loss: {avg_vloss:.4f}')
    return avg_vloss

# Training loop
EPOCHS = n_epochs
best_vloss = float('inf')

for epoch_index in range(EPOCHS):
    print(f'EPOCH {epoch_index + 1}:')

    avg_loss = train_one_epoch(epoch_index)
    avg_vloss = validate()
    scheduler.step(avg_vloss)
    writer.add_scalars('Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch_index + 1)
    writer.flush()

    """if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model_{timestamp}_epoch_{epoch_index + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved: {model_path}')"""

writer.close()

pl.seed_everything(42)
trainer = pl.Trainer(accelerator="auto", gradient_clip_val=0.01)
res = Tuner(trainer).lr_find(NBEATS, train_dataloaders=train_loader, min_lr=1e-5)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
NBEATS.hparams.learning_rate = res.suggestion()

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    enable_model_summary=True,
    enable_progress_bar = True,
    inference_mode = True,
    gradient_clip_val=0.01,
    callbacks=[early_stop_callback],
)

trainer.fit(
    NBEATS,
    train_dataloaders=train_loader,
    val_dataloaders=test_loader,
)
