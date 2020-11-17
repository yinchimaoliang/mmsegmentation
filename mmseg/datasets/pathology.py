import mmcv
from math import ceil
from os import path as osp

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
        self.use_patch = use_patch
        self.random_sampling = random_sampling
        self.horizontal_stride = horizontal_stride
        self.vertical_stride = vertical_stride
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_num = patch_num
        self.img_dict = dict()
        self.ann_dict = dict()
        super(PathologyDataset, self).__init__(
            img_suffix='.JPG', seg_map_suffix='_mask.png', **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = super().load_annotations(img_dir, img_suffix, ann_dir,
                                             seg_map_suffix, split)
        if self.use_patch:
            patch_infos = []
            for img_info in img_infos:
                filename = img_info['filename']
                seg_map_name = img_info['ann']['seg_map']
                img = mmcv.imread(osp.join(self.img_dir, filename))
                ann = mmcv.imread(
                    osp.join(self.ann_dir, seg_map_name), 'grayscale')
                self.img_dict[filename] = img
                self.ann_dict[filename] = ann
                if not self.random_sampling:
                    img_height = img.shape[0]
                    img_width = img.shape[1]
                    for i in range(
                            int(ceil(img_height / self.vertical_stride))):
                        for j in range(
                                int(ceil(img_width / self.horizontal_stride))):
                            if j * self.horizontal_stride + self.patch_width \
                                    < img_width:
                                left = j * self.horizontal_stride
                            else:
                                left = img_width - self.patch_width
                            if i * self.vertical_stride + self.patch_height \
                                    < img_height:
                                up = i * self.vertical_stride
                            else:
                                up = img_height - self.patch_height
                            img_info['img_prefix'] = self.img_dir
                            img_info['img'] = img
                            img_info['ann']['gt_semantic_seg'] = ann
                            img_info['patch_info'] = dict(
                                filename=filename,
                                up=up,
                                left=left,
                                patch_height=self.patch_height,
                                patch_width=self.patch_width)
                            patch_infos.append(img_info)
                else:
                    patch_infos.append(dict(filename=filename))
                return patch_infos

        return img_infos
