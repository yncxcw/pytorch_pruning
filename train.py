"""Entry point of training script."""

import argparse
import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import registered_datalaoders
from model import registered_models

class Trainer:
    """Trainer for training a model on a given dataset."""

    def __init__(self, epochs: int, learning_rate: float, train_dataloader, eval_dataloader, model: torch.nn.Module,
                 output: str, checkpoint: str):
        """
         Train on the dataset for 1 epoch.

         Args:
            epochs (int): Numbers of epoch to train.
            learning_rate (float): Learning rate.
            train_dataloader (Dataloader): Dataloader for training dataset.
            eval_dataloader (Dataloader): Dataloader for eval dataset.
            model (torch.nn.Module): Model to train.
            checkpoint (str): Path to a checkpoint file, used for evaluation.
        """
        self._epochs = epochs
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        self._model = model

        if checkpoint is not None:
            self._model.load_state_dict(
                torch.load(checkpoint)
            )
        # Copy model weights to gpu
        self._model.cuda()
        self._optimizer = optim.SGD(model.parameters(), lr=learning_rate,  momentum=0.9, weight_decay=5e-4)
        # TODO Why adam is not working
        # self._optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self._optimizer,
            # TODO: Hardcode for 200 epochs, fix this.
            milestones=[60, 120, 180],
            gamma=0.2,
            verbose=True,
        )
        self._log_dir = os.path.join(
            output, model.name(), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        os.makedirs(self._log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self._log_dir)

        # TODO: Currently hardcoded to cfar100 dataset.
        self._input_tensor = torch.zeros([1, 3, 32, 32], dtype=torch.float32).cuda()
        self._writer.add_graph(self._model, self._input_tensor)

    def train(self, epoch):
        """Train for one epoch."""
        # This has effects on layers like batchnorm or dropout.
        self._model.train()
        print(f"Start training for {len(self._train_dataloader)}")
        for index, (images, labels) in enumerate(self._train_dataloader):
            labels = labels.cuda()
            images = images.cuda()

            # If we don't clear grad from the last step, the model will learn nothing. 
            self._optimizer.zero_grad()        
            predictions = self._model(images)
            loss = self._loss_function(predictions, labels)
            loss.backward()
            self._optimizer.step()

            n_step = epoch * len(self._train_dataloader) + index
            self._writer.add_scalar("Train/Loss", loss.item(), n_step)
            self._writer.add_scalar("Train/LR", self._optimizer.param_groups[0]["lr"], n_step)
            # print(f"Training {n_step}/{self._epochs * len(self._train_dataloader)} Loss: {loss.item()}")

        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            self._writer.add_histogram("f{layer}/{attr[1:]}", param, epoch)

    @torch.no_grad()
    def evaluate(self, epoch):
        """Evaluate for one epoch.""" 
        print(f"Start validation for {len(self._eval_dataloader)}")
        # This has effects on layers like dropout and batchnorm
        self._model.eval()
        test_loss = 0 
        correct_1 = 0.0
        correct_5 = 0.0
        for index, (images, labels) in enumerate(self._eval_dataloader):
            labels = labels.cuda()
            images = images.cuda()

            predictions = self._model(images)
            loss = self._loss_function(predictions, labels)
            
            # Accumuate the loss over the mini batch            
            test_loss += loss.item()
            # The model output is a tensor with shape (N, C)

            # This choose the top5 prediction indices over the batch dimension
            # pred is with shape (N, 5)
            _, pred = predictions.topk(k=5, dim=1, largest=True, sorted=True)
            # Expand label from (N) -> (N, 5) 
            labels = labels.view(labels.size(0), -1).expand_as(pred)

            # With shape (N, 5), each line represents if each of the top 5 predicitons
            # if a correct prediction
            correct = pred.eq(labels)

            correct_1 += correct[:, :1].sum()
            correct_5 += correct[:, :5].sum()

        test_loss = test_loss / len(self._eval_dataloader.dataset)
        top1_error = 1 - correct_1.float() / len(self._eval_dataloader.dataset)
        top5_error = 1 - correct_5.float() / len(self._eval_dataloader.dataset)
        print(f"Test for epoch {epoch} average loss: {test_loss} top5_error: {top5_error} top1_error: {top1_error}")
        
        self._writer.add_scalar("Test/Loss", test_loss, epoch)
        self._writer.add_scalar("Test/top5_error", top5_error, epoch)
        self._writer.add_scalar("Test/top1_error", top1_error, epoch)
        
        return 1 - top1_error

    def train_loop(self):
        best_accuracy = 0
        best_model = 0
        for epoch in range(self._epochs):
            #TODO: Add learning rate scheduler
            self.train(epoch)
            accuracy = self.evaluate(epoch)
            print(f"Accuracy model epoch {epoch} accuracy {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = epoch
            # Save model for every 10 epoch
            if epoch % 10 == 0:
                # Save to checkpoint
                model_path = os.path.join(self._log_dir, "model-"+str(epoch))
                print(f"Saving model for epoch {epoch}")
                torch.save(self._model.state_dict(), model_path)

                # Export to onnx model 
                onnx_path = os.path.join(self._log_dir, "model-"+str(epoch)+".onnx")

                torch.onnx.export(
                    model=self._model,
                    args=self._input_tensor,
                    f=onnx_path,
                    export_params=True
                )
            self._lr_scheduler.step()
        print(f"Best model model-{best_model}")
          

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar100"], help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    parser.add_argument("--output", type=str, required=True, help="Output dir.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--checkpoint", type=str, help="Path to the eval model")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise ValueError("Dataset path {} doesn't exist".format(args.dataset_path))
    if not os.path.exists(args.output):
        raise ValueError("Output path {} doesn't exist".format(args.output))

    # Step 1: Build datalaoder
    dataloader_builder = registered_datalaoders[args.dataset]
    train_dataloader = dataloader_builder(path=args.dataset_path, type="train", batch_size=args.batch_size, shuffle=True) 
    eval_dataloader = dataloader_builder(path=args.dataset_path, type="test", batch_size=args.batch_size, shuffle=True)

    # Step2: Build model
    model_builder = registered_models[args.model]
    # TODO: The number of classes should depend on dataset, right now it is hardcodede for cifar100.
    model = model_builder(100, True)

    # Step3: Builder trainer and train
    trainer = Trainer(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        model=model,
        output=args.output,
        checkpoint=args.checkpoint,
    )

    if args.checkpoint is not None:
        trainer.evaluate(0)
    else:
        trainer.train_loop()

    print("Training completes!!!")
