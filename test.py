from utils import get_prefix, ModelAccess, evaluate
from data import InteractionData
from parse import get_args

args = get_args()
dataset = InteractionData(args)
model = ModelAccess.load_checkpoint(f'{args.ckpt}/{get_prefix(args)}' + '.pth').to(args.device)

Recall, NDCG, Precise, F1, ARP = evaluate(dataset, model, args)
