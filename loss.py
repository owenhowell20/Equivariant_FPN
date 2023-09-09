import torch


def evaluate_ll(args, model, test_loader):
    ### log likelihood estimation
    model.eval()
    lls = []
    clss = []
    for batch_idx, batch in enumerate(test_loader):
        batch = {k:v.to(args.device) for k,v in batch.items()}
        
        ### model probilities: [b,num_classes]
        outputs = model( batch['img'] )

        ### one hot encoding of classes, [b,num_classes]
        target = batch['cls'].cpu()
        
        loss = F.cross_entropy( outputs , target)



    lls = np.concatenate(lls)
    clss = np.concatenate(clss)

    per_class_ll = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_ll[args.class_names[i]] = lls[mask]

    np.save(os.path.join(args.fdir, f'eval_log_likelihood.npy'), per_class_ll)


def evaluate_error(args, model, test_loader):
    model.eval()
    errors = []
    clss = []
    for batch_idx, batch in enumerate(test_loader):
        batch = {k:v.to(args.device) for k,v in batch.items()}
        pred_rotmat = model.predict(batch['img'], batch['cls']).cpu()
        gt_rotmat = batch['rot'].cpu()
        err = rotation_error(pred_rotmat, gt_rotmat)
        errors.append(err.numpy())
        clss.append(batch['cls'].squeeze().cpu().numpy())

    errors = np.concatenate(errors)
    clss = np.concatenate(clss)

    per_class_err = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_err[args.class_names[i]] = errors[mask]

    np.save(os.path.join(args.fdir, 'eval.npy'), per_class_err)