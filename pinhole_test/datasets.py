import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_mo
import torch.utils.data as tData
from PIL import Image
import os

# Disable DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

class PinholeDataset(tData.Dataset): 
    """ Pinhole dataset management class
    """
    # constants 
    SPLIT_TEST = 0.2 # ratio of images to use for training
    SEED = 42
    # 
    def __init__(self, data_dir: str = "./dataset/Pinhole", type_in: str = "train", transforms=None):
        """Constructor
        Args:
        data_dir (str, optional): location of the dataset. Defaults to 'dataset/Pinhole'.
        type_in (str, optional): use 'train' for training, 'val' for validation and 'test' for testing set. Defaults to 'train'.
        transforms (optional): pointer to the function to use to pre-process the data. 
        """
        
        super().__init__()
        
        self._data_dir = data_dir
        
        # get the list of files in the directory
        filenames = os.listdir(self._data_dir)
        
        # split the filenames to get the ground truth
        gt = [float(f.split("_")[1].split("mm.")[0]) for f in filenames]
        
        # create a dataframe
        self._gtFr = pd.DataFrame({"image": filenames, "focal_length": gt})      
        
        # splt the dataset (with seed set, to have a constant)
        trainValIdxs, testIdxs = skl_mo.train_test_split(self._gtFr.index, test_size=self.SPLIT_TEST,
                                                        random_state=self.SEED)
         
        trainIdxs, valIdxs = skl_mo.train_test_split(self._gtFr.loc[trainValIdxs].index, test_size=self.SPLIT_TEST,
                                                    random_state=self.SEED)
        # filter 
        self._type = type_in
        if type_in == "train":
            self._dataFr = self._gtFr.loc[trainIdxs]
        elif type_in == "val":
            self._dataFr = self._gtFr.loc[valIdxs]
        elif type_in == "test":
            self._dataFr = self._gtFr.loc[testIdxs]
        else: 
            raise Exception("type_in must be train or test")
        
        self._transforms = transforms

    def __len__(self):
        return len(self._dataFr)
    
    def __getitem__(self, idx):
        img_path = self._data_dir + '/' + self._dataFr.iloc[idx]["image"]
        label = self._dataFr.iloc[idx]["focal_length"]
        
        image = Image.open(img_path)
        
        if self._transforms:
            image = self._transforms(image)
            
        return image, label
    
if __name__ == "__main__":
    # Test run with DataLoader wrapper
    dataset = PinholeDataset("dataset/Pinhole", "train")
    dataloader = tData.DataLoader(dataset, batch_size=1)
    
    # simulate a single iteration of a for loop
    img_batch, lbl_batch = next(iter(dataloader))
    
    # plot
    plt.figure("Image", (12, 6)) 
    plt.imshow(img_batch[0, :, :, :].permute(1, 2, 0))
    plt.show()