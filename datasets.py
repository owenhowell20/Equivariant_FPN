import torch
import torchvision

### for debugging
class placeholder_dataset(torch.utils.data.Dataset):
    def __init__(self, ):

        self.num_classes = 10 ### number of "classes"
        self.data = {
            'img' : torch.rand( 100, 3, 256, 256 ) ,
            'label' : torch.randint( self.num_classes , (100,) ) , } ### "data"
        self.class_names = ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet')

    def __getitem__(self, index):
        img = self.data['img'][index].to(torch.float32) / 255.

        if img.shape[0] != 3:
            img = img.expand(3,-1,-1)

        class_index = self.data['label'][index]

        return dict(img=img, label=class_index )

    @property
    def img_shape(self):
        return (3, 256, 256)

    def __len__(self):
        return len(self.data['img'])



def create_dataloaders(args):


	if args.dataset_name=='imagenet':

		### image dataset
		dataset = torchvision.datasets.ImageNet('data/imagenet')

		train_size = int(0.8 * len(dataset))
		test_size = len(dataset) - train_size
		train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

		args.img_shape = train_set.img_shape
		args.num_classes = train_set.num_classes

		print( f'{len(train_set)} train imgs; {len(test_set)} test imgs' )

		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers, drop_last=True)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

		return train_loader , test_loader , args


	if args.dataset_name=='placeholder':

		### image dataset
		dataset = placeholder_dataset()

		args.img_shape = dataset.img_shape
		args.num_classes = dataset.num_classes

		train_size = int(0.8 * len(dataset))
		test_size = len(dataset) - train_size
		train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

		print( f'{len(train_set)} train imgs; {len(test_set)} test imgs' )

		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers, drop_last=True)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

		return train_loader , test_loader , args





