# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TinyImageNet(CustomDataset):
    """TinyImageNet  Dataset.

    The dataset supports two kinds of annotation format. More details can be
    found in :class:`CustomDataset`.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use the default ImageNet-1k classes names.

            Defaults to None.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'Egyptian cat',
        'reel',
        'volleyball',
        'rocking chair, rocker',
        'lemon',
        'bullfrog, Rana catesbeiana',
        'basketball',
        'cliff, drop, drop-off',
        'espresso',
        'plunger, plumber\'s helper',
        'parking meter',
        'German shepherd, German shepherd dog, German police dog, alsatian',
        'dining table, board',
        'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
        'brown bear, bruin, Ursus arctos',
        'school bus',
        'pizza, pizza pie',
        'guinea pig, Cavia cobaya',
        'umbrella',
        'organ, pipe organ',
        'oboe, hautboy, hautbois',
        'maypole',
        'goldfish, Carassius auratus',
        'potpie',
        'hourglass',
        'seashore, coast, seacoast, sea-coast',
        'computer keyboard, keypad',
        'Arabian camel, dromedary, Camelus dromedarius',
        'ice cream, icecream',
        'nail',
        'space heater',
        'cardigan',
        'baboon',
        'snail',
        'coral reef',
        'albatross, mollymawk',
        'spider web, spider\'s web',
        'sea cucumber, holothurian',
        'backpack, back pack, knapsack, packsack, rucksack, haversack',
        'Labrador retriever',
        'pretzel',
        'king penguin, Aptenodytes patagonica',
        'sulphur butterfly, sulfur butterfly',
        'tarantula',
        'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
        'pop bottle, soda bottle',
        'banana',
        'sock',
        'cockroach, roach',
        'projectile, missile',
        'beer bottle',
        'mantis, mantid',
        'freight car',
        'guacamole',
        'remote control, remote',
        'European fire salamander, Salamandra salamandra',
        'lakeside, lakeshore',
        'chimpanzee, chimp, Pan troglodytes',
        'pay-phone, pay-station',
        'fur coat',
        'alp',
        'lampshade, lamp shade',
        'torch',
        'abacus',
        'moving van',
        'barrel, cask',
        'tabby, tabby cat',
        'goose',
        'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
        'bullet train, bullet',
        'CD player',
        'teapot',
        'birdhouse',
        'gazelle',
        'academic gown, academic robe, judge\'s robe',
        'tractor',
        'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
        'miniskirt, mini',
        'golden retriever',
        'triumphal arch',
        'cannon',
        'neck brace',
        'sombrero',
        'gasmask, respirator, gas helmet',
        'candle, taper, wax light',
        'desk',
        'frying pan, frypan, skillet',
        'bee',
        'dam, dike, dyke',
        'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
        'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
        'iPod',
        'punching bag, punch bag, punching ball, punchball',
        'beacon, lighthouse, beacon light, pharos',
        'jellyfish',
        'wok',
        'potter\'s wheel',
        'sandal',
        'pill bottle',
        'butcher shop, meat market',
        'slug',
        'hog, pig, grunter, squealer, Sus scrofa',
        'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
        'crane',
        'vestment',
        'dragonfly, darning needle, devil\'s darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk',
        'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
        'mushroom',
        'jinrikisha, ricksha, rickshaw',
        'water tower',
        'chest',
        'snorkel',
        'sunglasses, dark glasses, shades',
        'fly',
        'limousine, limo',
        'black stork, Ciconia nigra',
        'dugong, Dugong dugon',
        'sports car, sport car',
        'water jug',
        'suspension bridge',
        'ox',
        'ice lolly, lolly, lollipop, popsicle',
        'turnstile',
        'Christmas stocking',
        'broom',
        'scorpion',
        'wooden spoon',
        'picket fence, paling',
        'rugby ball',
        'sewing machine',
        'steel arch bridge',
        'Persian cat',
        'refrigerator, icebox',
        'barn',
        'apron',
        'Yorkshire terrier',
        'swimming trunks, bathing trunks',
        'stopwatch, stop watch',
        'lawn mower, mower',
        'thatch, thatched roof',
        'fountain',
        'black widow, Latrodectus mactans',
        'bikini, two-piece',
        'plate',
        'teddy, teddy bear',
        'barbershop',
        'confectionery, confectionary, candy store',
        'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
        'scoreboard',
        'orange',
        'flagpole, flagstaff',
        'American lobster, Northern lobster, Maine lobster, Homarus americanus',
        'trolleybus, trolley coach, trackless trolley',
        'drumstick',
        'dumbbell',
        'brass, memorial tablet, plaque',
        'bow tie, bow-tie, bowtie',
        'convertible',
        'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
        'orangutan, orang, orangutang, Pongo pygmaeus',
        'American alligator, Alligator mississipiensis',
        'centipede',
        'syringe',
        'go-kart',
        'brain coral',
        'sea slug, nudibranch',
        'cliff dwelling',
        'mashed potato',
        'viaduct',
        'military uniform',
        'pomegranate',
        'chain',
        'kimono',
        'comic book',
        'trilobite',
        'bison',
        'pole',
        'boa constrictor, Constrictor constrictor',
        'poncho',
        'bathtub, bathing tub, bath, tub',
        'grasshopper, hopper',
        'walking stick, walkingstick, stick insect',
        'Chihuahua',
        'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
        'lion, king of beasts, Panthera leo',
        'altar',
        'obelisk',
        'beaker',
        'bell pepper',
        'bannister, banister, balustrade, balusters, handrail',
        'bucket, pail',
        'magnetic compass',
        'meat loaf, meatloaf',
        'gondola',
        'standard poodle',
        'acorn',
        'lifeboat',
        'binoculars, field glasses, opera glasses',
        'cauliflower',
        'African elephant, Loxodonta africana',
    ]

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)
