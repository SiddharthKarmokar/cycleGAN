import torchvision.transforms as transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 10
BATCH_SIZE = 8
PHOTO_PATH = "../data/photo_jpg/"
MONET_PATH = "../data/monet_jpg/"
EPOCHES = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
