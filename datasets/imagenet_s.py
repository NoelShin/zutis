import os
from typing import Dict, List, Tuple
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from datasets.augmentations.geometric_transforms import resize
from utils.utils import getDistinctColors


class ImageNetSDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            n_categories: int,
            split: str = "val",
            device: torch.device = torch.device("cuda:0"),
    ):
        assert os.path.exists(dir_dataset), FileNotFoundError(dir_dataset)
        assert split in ["train", "val", "validation", "test"], ValueError(split)
        super(ImageNetSDataset, self).__init__()
        self.dir_dataset: str = dir_dataset
        self.device: torch.device = device
        self.split: str = split

        split = "validation" if split == "val" else split
        assert n_categories in [50, 300, 919], ValueError(n_categories)
        assert os.path.exists(dir_dataset)
        assert os.path.exists(f"{dir_dataset}/ImageNetS{n_categories}/{split}")

        self.p_images = sorted(glob(f"{dir_dataset}/ImageNetS{n_categories}/{split}/**/*.JPEG"))
        if split == "validation":
            self.p_gts: List[str] = sorted(
                glob(f"{dir_dataset}/ImageNetS{n_categories}/{split}-segmentation/**/*.png")
            )
            assert len(self.p_images) == len(self.p_gts), ValueError(f"{len(self.p_images)} != {len(self.p_gts)}")

        elif split == "train":
            self.p_gts: List[str] = sorted(
                glob(f"{dir_dataset}/ImageNetS{n_categories}/{split}-semi-segmentation/**/*.png")
            )
            assert len(self.p_images) == len(self.p_gts), ValueError(f"{len(self.p_images)} != {len(self.p_gts)}")

        else:
            print("No groundtruth masks are given for a test split.")

        self.n_categories: int = n_categories + 1  # + 1 for background

        assert len(self.p_images) > 0, ValueError("No images are found.")

        self.ignore_index: int = 1000

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.name = f"imagenet-s{n_categories}"
        self.max_size: int = 1024  # 640 if self.split == "val" else 1024  # 640 for validation split
        print(f"ImageNet-S{n_categories} ({split}) is loaded ({len(self.p_images)} images).")

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index: int) -> dict:
        dict_data: dict = {}

        p_image: str = self.p_images[index]

        image: Image.Image = Image.open(p_image).convert("RGB")

        # if self.n_categories == 920:
        W, H = image.size
        max_size = np.maximum(H, W).item()
        if max_size > self.max_size:
            image: Image.Image = resize(
                image, size=self.max_size, edge="longer", interpolation="bilinear"
            )
            # gt: Image.Image = resize(
            #     gt, size=self.max_size, edge="longer", interpolation="nearest"
            # )

        # normalise the image
        image: torch.Tensor = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        dict_data.update({
            "image": image,
            "p_image": p_image,
            "original_size": (H, W)
        })
        if self.split != "test":
            # gt: H x W x 3
            p_gt: str = self.p_gts[index]
            gt: Image.Image = Image.open(p_gt)
            gt: np.ndarray = np.array(gt).astype(np.int64)
            gt = gt[..., 0] + gt[..., 1] * 256
            gt: torch.Tensor = torch.from_numpy(gt).to(torch.int64)
            dict_data.update({
                "semantic_mask": gt,
                "p_gt": p_gt
            })
        return dict_data

imagenet_s_categories_50 = ['goldfish', 'tiger shark', 'goldfinch', 'tree frog', 'kuvasz', 'red fox', 'siamese cat', 'american black bear', 'ladybug', 'sulphur butterfly', 'wood rabbit', 'hamster', 'wild boar', 'gibbon', 'african elephant', 'giant panda', 'airliner', 'ashcan', 'ballpoint', 'beach wagon', 'boathouse', 'bullet train', 'cellular telephone', 'chest', 'clog', 'container ship', 'digital watch', 'dining table', 'golf ball', 'grand piano', 'iron', 'lab coat', 'mixing bowl', 'motor scooter', 'padlock', 'park bench', 'purse', 'streetcar', 'table lamp', 'television', 'toilet seat', 'umbrella', 'vase', 'water bottle', 'water tower', 'yawl', 'street sign', 'lemon', 'carbonara', 'agaric']
imagenet_s_categories_300 = ['tench', 'goldfish', 'tiger shark', 'hammerhead', 'electric ray', 'ostrich', 'goldfinch', 'house finch', 'indigo bunting', 'kite', 'common newt', 'axolotl', 'tree frog', 'tailed frog', 'mud turtle', 'banded gecko', 'american chameleon', 'whiptail', 'african chameleon', 'komodo dragon', 'american alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'king snake', 'rock python', 'horned viper', 'harvestman', 'scorpion', 'garden spider', 'tick', 'african grey', 'lorikeet', 'red-breasted merganser', 'wallaby', 'koala', 'jellyfish', 'sea anemone', 'conch', 'fiddler crab', 'american lobster', 'spiny lobster', 'isopod', 'bittern', 'crane bird', 'limpkin', 'bustard', 'albatross', 'toy terrier', 'afghan hound', 'bluetick', 'borzoi', 'irish wolfhound', 'whippet', 'ibizan hound', 'staffordshire bullterrier', 'border terrier', 'yorkshire terrier', 'lakeland terrier', 'giant schnauzer', 'standard schnauzer', 'scotch terrier', 'lhasa', 'english setter', 'clumber', 'english springer', 'welsh springer spaniel', 'kuvasz', 'kelpie', 'doberman', 'miniature pinscher', 'malamute', 'pug', 'leonberg', 'great pyrenees', 'samoyed', 'brabancon griffon', 'Cardigan Welsh corgi', 'coyote', 'red fox', 'kit fox', 'grey fox', 'persian cat', 'siamese cat', 'cougar', 'lynx', 'tiger', 'american black bear', 'sloth bear', 'ladybug', 'leaf beetle', 'weevil', 'bee', 'cicada', 'leafhopper', 'damselfly', 'ringlet', 'cabbage butterfly', 'sulphur butterfly', 'sea cucumber', 'wood rabbit', 'hare', 'hamster', 'wild boar', 'hippopotamus', 'bighorn', 'ibex', 'badger', 'three-toed sloth', 'orangutan', 'gibbon', 'colobus', 'spider monkey', 'squirrel monkey', 'madagascar cat', 'indian elephant', 'african elephant', 'giant panda', 'barracouta', 'eel', 'coho', 'academic gown', 'accordion', 'airliner', 'ambulance', 'analog clock', 'ashcan', 'backpack', 'balloon', 'ballpoint', 'barbell', 'barn', 'bassoon', 'bath towel', 'beach wagon', 'bicycle-built-for-two', 'binoculars', 'boathouse', 'bonnet', 'bookcase', 'bow', 'brass', 'breastplate', 'bullet train', 'cannon', 'can opener', "carpenter's kit", 'cassette', 'cellular telephone', 'chain saw', 'chest', 'china cabinet', 'clog', 'combination lock', 'container ship', 'corkscrew', 'crate', 'crock pot', 'digital watch', 'dining table', 'dishwasher', 'doormat', 'dutch oven', 'electric fan', 'electric locomotive', 'envelope', 'file', 'folding chair', 'football helmet', 'freight car', 'french horn', 'fur coat', 'garbage truck', 'goblet', 'golf ball', 'grand piano', 'half track', 'hamper', 'hard disc', 'harmonica', 'harvester', 'hook', 'horizontal bar', 'horse cart', 'iron', "jack-o'-lantern", 'lab coat', 'ladle', 'letter opener', 'liner', 'mailbox', 'megalith', 'military uniform', 'milk can', 'mixing bowl', 'monastery', 'mortar', 'mosquito net', 'motor scooter', 'mountain bike', 'mountain tent', 'mousetrap', 'necklace', 'nipple', 'ocarina', 'padlock', 'palace', 'parallel bars', 'park bench', 'pedestal', 'pencil sharpener', 'pickelhaube', 'pillow', 'planetarium', 'plastic bag', 'polaroid camera', 'pole', 'pot', 'purse', 'quilt', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'reflex camera', 'refrigerator', 'rifle', 'rocking chair', 'rubber eraser', 'rule', 'running shoe', 'sewing machine', 'shield', 'shoji', 'ski', 'ski mask', 'slot', 'soap dispenser', 'soccer ball', 'sock', 'soup bowl', 'space heater', 'spider web', 'spindle', 'sports car', 'steel arch bridge', 'stethoscope', 'streetcar', 'submarine', 'swimming trunks', 'syringe', 'table lamp', 'tank', 'teddy', 'television', 'throne', 'tile roof', 'toilet seat', 'trench coat', 'trimaran', 'typewriter keyboard', 'umbrella', 'vase', 'volleyball', 'wardrobe', 'warplane', 'washer', 'water bottle', 'water tower', 'whiskey jug', 'wig', 'wine bottle', 'wok', 'wreck', 'yawl', 'yurt', 'street sign', 'traffic light', 'consomme', 'ice cream', 'bagel', 'cheeseburger', 'hotdog', 'mashed potato', 'spaghetti squash', 'bell pepper', 'cardoon', 'granny smith', 'strawberry', 'lemon', 'carbonara', 'burrito', 'cup', 'coral reef', "yellow lady's slipper", 'buckeye', 'agaric', 'gyromitra', 'earthstar', 'bolete']
imagenet_s_categories_919 = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'european fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana', 'american chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'gila monster', 'green lizard', 'african chameleon', 'komodo dragon', 'african crocodile', 'american alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'rock python', 'indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge', 'african grey', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'american lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'american egret', 'bittern', 'crane bird', 'limpkin', 'european gallinule', 'american coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'chihuahua', 'japanese spaniel', 'maltese dog', 'pekinese', 'shih-tzu', 'blenheim spaniel', 'papillon', 'toy terrier', 'rhodesian ridgeback', 'afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan coonhound', 'walker hound', 'english foxhound', 'redbone', 'borzoi', 'irish wolfhound', 'italian greyhound', 'whippet', 'ibizan hound', 'norwegian elkhound', 'otterhound', 'saluki', 'scottish deerhound', 'weimaraner', 'staffordshire bullterrier', 'american staffordshire terrier', 'bedlington terrier', 'border terrier', 'kerry blue terrier', 'irish terrier', 'norfolk terrier', 'norwich terrier', 'yorkshire terrier', 'wire-haired fox terrier', 'lakeland terrier', 'sealyham terrier', 'airedale', 'cairn', 'australian terrier', 'dandie dinmont', 'boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'scotch terrier', 'tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 'west highland white terrier', 'lhasa', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'labrador retriever', 'chesapeake bay retriever', 'german short-haired pointer', 'vizsla', 'english setter', 'irish setter', 'gordon setter', 'brittany spaniel', 'clumber', 'english springer', 'welsh springer spaniel', 'cocker spaniel', 'sussex spaniel', 'irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'old english sheepdog', 'shetland sheepdog', 'collie', 'border collie', 'bouvier des flandres', 'rottweiler', 'german shepherd', 'doberman', 'miniature pinscher', 'greater swiss mountain dog', 'bernese mountain dog', 'appenzeller', 'entlebucher', 'boxer', 'bull mastiff', 'tibetan mastiff', 'french bulldog', 'great dane', 'saint bernard', 'eskimo dog', 'malamute', 'siberian husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland', 'great pyrenees', 'samoyed', 'pomeranian', 'chow', 'keeshond', 'brabancon griffon', 'pembroke', 'Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'mexican hairless', 'timber wolf', 'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'african hunting dog', 'hyena', 'red fox', 'kit fox', 'arctic fox', 'grey fox', 'tabby', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'american black bear', 'ice bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'angora', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'arabian camel', 'llama', 'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi', 'spider monkey', 'squirrel monkey', 'madagascar cat', 'indri', 'indian elephant', 'african elephant', 'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho', 'rock beauty', 'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'ambulance', 'amphibian', 'analog clock', 'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'balloon', 'ballpoint', 'band aid', 'banjo', 'barbell', 'barber chair', 'barn', 'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bath towel', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin', 'beer bottle', 'beer glass', 'bib', 'bicycle-built-for-two', 'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie', 'bonnet', 'bookcase', 'bow', 'bow tie', 'brass', 'brassiere', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'bullet train', 'cab', 'caldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan sweater', 'car mirror', "carpenter's kit", 'carton', 'cassette', 'cassette player', 'castle', 'catamaran', 'cello', 'cellular telephone', 'chain', 'chainlink fence', 'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'christmas stocking', 'church', 'cleaver', 'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'combination lock', 'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'tower crane', 'crash helmet', 'crate', 'crib', 'crock pot', 'croquet ball', 'crutch', 'cuirass', 'desk', 'dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag', 'dishwasher', 'doormat', 'drilling platform', 'drum', 'drumstick', 'dumbbell', 'dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'envelope', 'espresso maker', 'face powder', 'feather boa', 'file', 'fireboat', 'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain pen', 'four-poster', 'freight car', 'french horn', 'frying pan', 'fur coat', 'garbage truck', 'gasmask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand piano', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower', 'hand-held computer', 'handkerchief', 'hard disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'honeycomb', 'hook', 'hoopskirt', 'horizontal bar', 'horse cart', 'hourglass', 'ipod', 'iron', "jack-o'-lantern", 'jean', 'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lawn mower', 'lens cap', 'letter opener', 'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick', 'loafer', 'lotion', 'loudspeaker', 'loupe', 'magnetic compass', 'mailbag', 'mailbox', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'model t', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter', 'mountain bike', 'mountain tent', 'mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'pay-phone', 'pedestal', 'pencil box', 'pencil sharpener', 'perfume', 'petri dish', 'photocopier', 'pick', 'pickelhaube', 'picket fence', 'pickup', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack', 'plunger', 'polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 'pop bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projector', 'puck', 'punching bag', 'purse', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator', 'remote control', 'revolver', 'rifle', 'rocking chair', 'rubber eraser', 'rugby ball', 'rule', 'running shoe', 'safe', 'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale', 'school bus', 'schooner', 'scoreboard', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'slot', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'space heater', 'space shuttle', 'spatula', 'speedboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher', 'studio couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'suspension bridge', 'swab', 'sweatshirt', 'swimming trunks', 'swing', 'syringe', 'table lamp', 'tank', 'teapot', 'teddy', 'television', 'tennis ball', 'thatch', 'theater curtain', 'thimble', 'thresher', 'throne', 'tile roof', 'toaster', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'tractor', 'trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'windsor tie', 'wine bottle', 'wok', 'wooden spoon', 'worm fence', 'wreck', 'yawl', 'yurt', 'comic book', 'street sign', 'traffic light', 'menu', 'plate', 'guacamole', 'consomme', 'trifle', 'ice cream', 'ice lolly', 'french loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'granny smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'cup', 'eggnog', 'bubble', 'cliff', 'coral reef', 'ballplayer', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'hip', 'buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']

label_id_to_category_50 = {0: "background"}
label_id_to_category_50.update({
    label_id: category for label_id, category in enumerate(imagenet_s_categories_50, start=1)
})

label_id_to_category_300 = {0: "background"}
label_id_to_category_300.update({
    label_id: category for label_id, category in enumerate(imagenet_s_categories_300, start=1)
})

label_id_to_category_919 = {0: "background"}
label_id_to_category_919.update({
    label_id: category for label_id, category in enumerate(imagenet_s_categories_919, start=1)
})

list_colours_50 = getDistinctColors(50)
list_colours_50.insert(0, (0, 0, 0))  # insert a black colour for a "background" category
imagenet_s50_palette: Dict[int, Tuple[float, float, float]] = {
    label_id: colour for label_id, colour in enumerate(list_colours_50)
}
imagenet_s50_palette[1000] = (255, 255, 255)

list_colours_300 = getDistinctColors(300)
list_colours_300.insert(0, (0, 0, 0))  # insert a black colour for a "background" category
imagenet_s300_palette: Dict[int, Tuple[float, float, float]] = {
    label_id: colour for label_id, colour in enumerate(list_colours_300)
}
imagenet_s300_palette[1000] = (255, 255, 255)

list_colours_919 = getDistinctColors(919)
list_colours_919.insert(0, (0, 0, 0))  # insert a black colour for a "background" category
imagenet_s919_palette: Dict[int, Tuple[float, float, float]] = {
    label_id: colour for label_id, colour in enumerate(list_colours_919)
}
imagenet_s919_palette[1000] = (255, 255, 255)
