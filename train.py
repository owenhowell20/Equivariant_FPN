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
from datasets import create_dataloaders
from model import FPN_predictor

### check that cuda is avalible:
print()
print( "Cuda is available:" , torch.cuda.is_available() )
print()

warnings.filterwarnings( 'ignore', category=UserWarning )

### TO DO: 
### 1. make accuracy mesure output for each class
### 2. transforms on all datasets
### 3. fix dataloaders


### create equivarient FPN model
def create_model(args):

    model = FPN_predictor( so2_gspace=args.so2_gspace, num_classes=args.num_classes ).to(args.device)

    num_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    print( 'total number of model parameters:' , num_params )
    model.train()

    return model


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    fname = 'test' ###f"_{args.dataset_name}_{args.encoder.replace('_','-')}_seed{args.seed}"

    #if args.desc != '':
    #    fname += f"_{args.desc}"
    args.fdir = os.path.join(args.results_dir, fname)
  
    if not os.path.exists(args.fdir):
        os.makedirs(args.fdir)

    with open(os.path.join(args.fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))


    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers =  [logging.StreamHandler(), logging.FileHandler(os.path.join(args.fdir, "log.txt"))]


    train_loader, test_loader, args = create_dataloaders(args)

    model = create_model(args)
    optimizer = torch.optim.SGD( model.parameters(), lr=args.lr_initial, momentum=args.sgd_momentum, weight_decay=args.weight_decay, nesterov=bool(args.use_nesterov), )
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
        model.train()
    else:
        starting_epoch = 1

    data = []
    for epoch in range(starting_epoch, args.num_epochs+1):
        train_loss = 0
        train_acc = []
        time_before_epoch = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k:v.to(args.device) for k,v in batch.items()}
            
            loss, acc = model.compute_loss(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc.append(acc)

        train_loss /= batch_idx + 1
        train_acc_median = np.median(train_acc)

        test_loss = 0
        test_acc = []
        test_cls = []

        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = {k:v.to(args.device) for k,v in batch.items()}
            with torch.no_grad():
                loss, acc = model.compute_loss(**batch)

            test_loss += loss.item()
            test_acc.append(acc)
            #test_cls.append(batch['cls'].squeeze(0).cpu().numpy())

        model.train()
        test_loss /= batch_idx + 1
        test_acc_median = np.median(test_acc)

        per_class_err = {}
        # test_acc = np.array(test_acc).flatten()
        # test_cls = np.array(test_cls).flatten()
        # for i, cls in enumerate(args.class_names):
        #     per_class_err[cls] = f"{np.degrees(np.median(test_acc[test_cls == i])):.1f}"

        logger.info(str(per_class_err))

        data.append(dict(epoch=epoch,
                         time_elapsed=time.perf_counter() - time_before_epoch,
                         train_loss=train_loss,
                         test_loss=test_loss,
                         train_acc_median=train_acc_median,
                         test_acc_median=test_acc_median,
                         lr=optimizer.param_groups[0]['lr'],
                        ))
        lr_scheduler.step()

        ### checkpointing
        torch.save( {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'done': False,
                   }, os.path.join(args.fdir, "checkpoint.pt") )

        log_str = 'test_2' ###f"Epoch {epoch}/{args.num_epochs} | " + f"LOSS={train_loss:.4f}<{test_loss:.4f}> " + f"ROT ERR={np.degrees(test_acc_median):.2f}° | " + f"time={time.perf_counter() - time_before_epoch:.1f}s | " + f"lr={lr_scheduler.get_last_lr()[0]:.1e}"
        logger.info(log_str)
        time_before_epoch = time.perf_counter()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--desc', type=str, default='')

	### model architecture params:
	parser.add_argument('--so2_gspace', type=int, default=4, help='Discretization of SO(2) Group')
	parser.add_argument('--encoder', type=str, default='eqv_fpn', choices=[ 'fpn', 'eqv_fpn' ] , help='Choice of Network Head')

	### training params:
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr_initial', type=float, default=0.001)
	parser.add_argument('--lr_step_size', type=int, default=15)
	parser.add_argument('--lr_decay_rate', type=float, default=0.1)
	parser.add_argument('--sgd_momentum', type=float, default=0.9)
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



