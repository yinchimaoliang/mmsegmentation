from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PathologyDataset(CustomDataset):
    """Pathology dataset.

    The ``img_suffix`` is fixed to '.JPG' and ``seg_map_suffix``
        is fixed to '_mask.png' for Pathology dataset.

    Args:
        use_patch (bool): Whether to use patch. If true, the whole
            image will be cropped into patches. Default: True.
        random_sampling (bool): Whether to crop the patches randomly,
            only works if use_patch is true. Default: False.
        horizontal_stride (int): Horizontal stride of the patch,
            only works if use_patch is true and random_sampling is false.
            Default: 512.
        horizontal_stride (int): vertical stride of the patch,
            only works if use_patch is true and random_sampling is false.
            Default: 512.
        patch_width (int): Width of the patch, only works if use_patch is
            true and random_sampling is false. Default: 512.
        patch_height (int): Height of the patch, only works
            if use_patch is true and random_sampling is false.
            Default: 512.
        patch_num (int): Number of patches to crop for one image, only
            works if use_patch is true and random_sampling is true.
            Default : 1.
    """

    CLASSES = ('background', 'inflammation', 'low', 'high', 'carcinoma')

    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]]

    def __init__(self,
                 use_patch=True,
                 random_sampling=False,
                 horizontal_stride=512,
                 vertical_stride=512,
                 patch_width=512,
                 patch_height=512,
                 patch_num=1,
                 **kwargs):
        super(PathologyDataset, self).__init__(
            img_suffix='.JPG', seg_map_suffix='_mask.png', **kwargs)
        self.use_patch = use_patch
        self.random_sampling = random_sampling,
        self.horizontal_stride = horizontal_stride,
        self.vertical_stride = vertical_stride
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_num = patch_num
