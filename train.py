import torch.optim as optim
from torch.utils.data import DataLoader

from utils import BPRLoss, set_seed
from data import InteractionData
from model import MF, DICE
from processor import Processor
from parse import get_args


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    dataset = InteractionData(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True)

    if args.model == 'DICE':
        model = DICE(dataset.num_user, dataset.num_item, args.dim).to(args.device)
    else:
        model = MF(dataset.num_user, dataset.num_item, args.dim).to(args.device)
    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    processor = Processor(model, criterion, optimizer, dataloader, dataset, args)
    processor.process()
