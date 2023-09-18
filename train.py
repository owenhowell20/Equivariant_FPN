### stolen from David Klee, Image to Sphere
import torch
import re
import argparse
import os
import time
from datetime import datetime
import logging
import numpy as np
import torch.nn.functional as F
import torch
import warnings
import wandb

from datasets import create_dataloaders
from model import FPN_predictor

### check that cuda is avalible:
print()
print( "Cuda is available:" , torch.cuda.is_available() )
print()

warnings.filterwarnings( 'ignore', category=UserWarning )

### TO DO: (In order of importance)
### 1. write imagenet dataloader and coco dataloader
### 2. write in wandb gradients check
### 3. make accuracy mesure+confusion matrix output for each class, also need to get class names from file
### 4. add multi-gpu support
### 5. Maybe need to add some semi-supervised preprocessing step: What should the latent space be? Add a decoder module that is later discarded
### 6. Check how balanced dataset is: compute the random cross entropy.
### 7. Maybe can change heads to be of resnet form
### 8. Should last conv be to invarient features?

### create equivarient FPN model
def create_model(args):

    model = FPN_predictor( so2_gspace=args.so2_gspace, num_classes=args.num_classes , encoder = args.encoder , recombinator=args.recombinator ).to(args.device)

    num_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    print( 'total number of model parameters:' , num_params )
    model.train()

    return model


def main(args):

    ###
    wandb.init(
    # set the wandb project where this run will be logged
    project="Equivariant_FPN",
    # track hyperparameters and run metadata
    config={
    "dataset_name": args.dataset_name,
    "learning_rate": args.lr_initial,
    "encoder": args.encoder,
    "epochs": args.num_epochs,
    "recombinator": args.recombinator,
    "num_workers": args.num_workers,
    "batch_size": args.batch_size,
    "lr_step_size": args.lr_step_size,
    "lr_decay_rate": args.lr_decay_rate,
    "so2_gspace": args.so2_gspace,
    "num_epochs":args.num_epochs,
    "training_method":args.optimizer,
    }
    )    

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    ### file to save to
    fname = f"_{args.dataset_name}_{args.encoder.replace('_','-')}_seed{args.seed}_{args.recombinator.replace('_','-')}_gspace{args.so2_gspace}_method{args.optimizer}"
    if args.desc != '':
       fname += f"_{args.desc}"
    args.fdir = os.path.join(args.results_dir, fname)
  
    if not os.path.exists(args.fdir):
        os.makedirs(args.fdir)

    with open(os.path.join(args.fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))


    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers =  [ logging.StreamHandler(), logging.FileHandler(os.path.join(args.fdir, "log.txt")) ]

    train_loader, test_loader, args = create_dataloaders(args)

    model = create_model(args)

    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD( model.parameters(), lr=args.lr_initial, momentum=args.sgd_momentum, weight_decay=args.weight_decay, nesterov=bool(args.use_nesterov), )
    
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam( model.parameters(), lr=args.lr_initial, weight_decay=args.weight_decay  )

    ### maybe should use alternating?
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_decay_rate)

    if os.path.exists(os.path.join(args.fdir, "checkpoint.pt")):
        # read the log to find the epoch
        checkpoint = torch.load(os.path.join(args.fdir, "checkpoint.pt"))
        if checkpoint['done']:
            exit()

        starting_epoch = checkpoint['epoch'] + 1
        epoch = starting_epoch
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("Model Loaded From File!")
        model.train()
    else:
        starting_epoch = 1

    data = []
    for epoch in range(starting_epoch, args.num_epochs+1):
        train_loss = 0
        train_acc = []
        time_before_epoch = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            batch = { k:v.to(args.device) for k,v in batch.items() }
            
            loss, num_correct, preds = model.compute_loss(**batch)
            per_correct = num_correct/args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc.append( per_correct.cpu().numpy() )

        train_loss /= batch_idx + 1
        train_acc_median = np.median(train_acc)

        test_loss = 0
        test_acc = []
        test_cls = []

        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = {k:v.to(args.device) for k,v in batch.items()}
            with torch.no_grad():
                loss, num_correct, preds = model.compute_loss(**batch)
                per_correct = num_correct/args.batch_size

            test_loss += loss.item()
            test_acc.append( per_correct.cpu().numpy() )

        model.train()
        test_loss /= batch_idx + 1
        test_acc_median = np.median(test_acc)

        per_class_err = {}
        test_acc = np.array(test_acc).flatten()
        logger.info( str(test_acc) )

        data.append( dict(epoch=epoch,
                         time_elapsed=time.perf_counter() - time_before_epoch,
                         train_loss=train_loss,
                         test_loss=test_loss,
                         train_acc_median=train_acc_median,
                         test_acc_median=test_acc_median,
                         lr=optimizer.param_groups[0]['lr'],
                        ) )
        lr_scheduler.step()

        ### wandb log
        wandb.log(dict(epoch=epoch,
                         time_elapsed=time.perf_counter() - time_before_epoch,
                         train_loss=train_loss,
                         test_loss=test_loss,
                         train_acc_median=train_acc_median,
                         test_acc_median=test_acc_median,
                         current_lr=optimizer.param_groups[0]['lr'],
                        ))

        ### checkpointing
        torch.save( {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'done': False,
                   }, os.path.join(args.fdir, "checkpoint.pt") )

        log_str = f"Epoch {epoch}/{args.num_epochs} | " \
                  + f"LOSS={train_loss:.4f}<{test_loss:.4f}> " \
                  + f"Percentage Correct={test_acc_median:.2f} | " \
                  + f"time={time.perf_counter() - time_before_epoch:.1f}s | " \
                  + f"lr={lr_scheduler.get_last_lr()[0]:.1e}"

        logger.info(log_str)
        time_before_epoch = time.perf_counter()

    ### save final trained model
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'done' : True,
               }, os.path.join(args.fdir, "checkpoint.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--desc', type=str, default='')

    parser.add_argument('--encoder', type=str, default='eqv_fpn210', choices=[ 'fpn', 'eqv_fpn101', 'eqv_fpn210' ] , help='Choice of FPN Head')
    parser.add_argument('--recombinator', type=str, default='quorum', choices=[ 'concat' , 'quorum' , 'attention' ] , help='Choice of feature recombination module')
    parser.add_argument('--so2_gspace', type=int, default=4, help='Discretization of SO(2) Group')

    parser.add_argument('--optimizer', type=str, default='sgd', choices = ['sgd', 'adam'] , help='Choice of optimizer' )
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_initial', type=float, default=0.01)
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--lr_decay_rate', type=float, default=0.05)
    parser.add_argument('--sgd_momentum', type=float, default=0.8 )
    parser.add_argument('--use_nesterov', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)

    ### dataset and results info
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--dataset_name', type=str, default='caltech101', choices=['imagenet',  'caltech101', 'caltech256' , 'coco' , 'placeholder' ] )

    ### number of workers used
    parser.add_argument('--num_workers', type=int, default=4, help='workers used by dataloader')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)



