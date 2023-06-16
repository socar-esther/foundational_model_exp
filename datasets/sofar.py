import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

## 클래스 추가
NEW_CNAMES = {
    "01_outer_normal": "a clean white car", # 1
    "02_outer_damage": "a white car that has been damaged", # 2
    "03_outer_dirt" : "a dirt white car", # 3
    "04_outer_wash": "a white car being washed", # 4
    "05_inner_wash": "the window of a car while being washed", # 5
    "06_inner_dashboard" : "the dashboard of a car", # 6
    "07_inner_cupholder" : "a clean cupholder of a car", # 7
    "08_inner_cupholder_dirt" : "a dirty cupholder of a car", # 8
    "09_inner_glovebox" : "the glovebox of a car", # 9
    "10_inner_washer_fluid" : "a water fluid bottle", # 10
    "11_inner_front_seat" : "a front seat of a car", # 11
    "12_inner_rear_seat" : "the back seat of a car", # 12
    "13_inner_trunk" : "a car trunk", # 13
    "14_inner_sheet_clean " : "a clean car floor", # 14
    "15_inner_sheet_dirt" : "a dirty car floor", # 15
    "16_inner_seat_dirt" : "dirty seat in a car", # 16
    "17_outer_rainy": "a car with rain drop", # 17
    "18_outer_snowy" : "a white car covered in snow", # 18
    "19_outer_tire" : "the tire of a white car" # 19
}

@DATASET_REGISTRY.register()
class SoFAR(DatasetBase):

    dataset_dir = "sofar"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SOFAR.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        
        print('> check splitted dataset path:', self.split_path)
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES) # 새로운 caption 사용
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        num_shots = 2000 # TODO: sofar에 넣을 shot수 지정
        
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # few shot setting에서 새로운 class를 지정해서 나누는 경우
        #subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        #train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
