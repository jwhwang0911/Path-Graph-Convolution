from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


# parameter : Dataset, batch_size, shuffle = {True or False}, num_workers, pin_memory = {True or False}
class CustomDataLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__)