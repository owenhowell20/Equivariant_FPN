import torch
import torchvision


def create_dataloaders(args):

	### image dataset
	imagenet_data = torchvision.datasets.ImageNet('/scratch/howell.o/imagenet_root/')

	train_size = int(0.8 * len(full_dataset))
	test_size = len(full_dataset) - train_size
	train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

	args.img_shape = train_set.img_shape
    args.num_classes = train_set.num_classes
    args.class_names = train_set.class_names

    print( f'{len(train_set)} train imgs; {len(test_set)} test imgs' )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)


    return train_loader, test_loader, args




if __name__ == "__main__":

	
	so2_gspace = 4
	num_classes = 64
	batch_size = 4

	x = torch.rand( batch_size , 3 , 256 , 256 )

	f = FPN_predictor( so2_gspace, num_classes )

	### unchanged y-values:
	y = f( x )

	print( y[0].shape , y[1].shape, y[2].shape , y[3].shape )
	quit()

	### check for so2-invarience of outputs:
	for g in so2.elements:

		x_rot = x.transform(g)

		### new inputs
		y_rot = f( x_rot.tensor )

		# ### meausre the differences:
		z0 = y[0].transform(g)
		z1 = y[1].transform(g)
		z2 = y[2].transform(g)
		z3 = y[3].transform(g)


		### mesure differences
		d0 = z0.tensor - y_rot[0].tensor
		d1 = z1.tensor - y_rot[1].tensor
		d2 = z2.tensor - y_rot[2].tensor
		d3 = z3.tensor - y_rot[3].tensor

		### take the norm
		print()
		print("group element:" , g)
		print( 'zero percentage error:' ,  torch.norm(d0)/torch.norm( z0.tensor ) ) 
		print( 'one percentage error:' ,  torch.norm(d1)/torch.norm( z1.tensor ) ) 
		print( 'two percentage error:' ,  torch.norm(d2)/torch.norm( z2.tensor ) ) 
		print( 'three percentage error:' ,  torch.norm(d3)/torch.norm( z3.tensor ) ) 
		print()

	### check types of outputs
	###print( y_rot[0].type , y_rot[1].type , y_rot[2].type , y_rot[3].type )

