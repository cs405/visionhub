# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import numpy as np
from typing import List, Dict


class PersonAttribute(object):
    """行人属性识别后处理

    支持 26 个行人属性:
    - 性别、年龄、方向
    - 眼镜、帽子、手持物
    - 包类型、上衣、下衣、鞋子

    Args:
        threshold (float): 通用阈值，默认 0.5
        glasses_threshold (float): 眼镜阈值，默认 0.3
        hold_threshold (float): 手持物阈值，默认 0.6
    """

    def __init__(
        self,
        threshold: float = 0.5,
        glasses_threshold: float = 0.3,
        hold_threshold: float = 0.6
    ):
        self.threshold = threshold
        self.glasses_threshold = glasses_threshold
        self.hold_threshold = hold_threshold

    def __call__(self, batch_preds, file_names=None) -> List[Dict]:
        """后处理

        Args:
            batch_preds: shape [B, 26]
            file_names: 文件名列表

        Returns:
            结果列表，每个元素:
            {
                'attributes': ['Male', 'Age18-60', ...],
                'output': [0, 1, 0, ...]  # 二值化输出
            }
        """
        # 属性定义
        age_list = ['AgeLess18', 'Age18-60', 'AgeOver60']
        direct_list = ['Front', 'Side', 'Back']
        bag_list = ['HandBag', 'ShoulderBag', 'Backpack']
        upper_list = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        lower_list = [
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts',
            'Skirt&Dress'
        ]

        batch_res = []
        for res in batch_preds:
            res = res.tolist() if isinstance(res, np.ndarray) else res
            label_res = []

            # 性别 (索引 22)
            gender = 'Female' if res[22] > self.threshold else 'Male'
            label_res.append(gender)

            # 年龄 (索引 19-21)
            age = age_list[np.argmax(res[19:22])]
            label_res.append(age)

            # 方向 (索引 23-25)
            direction = direct_list[np.argmax(res[23:])]
            label_res.append(direction)

            # 眼镜 (索引 1)
            glasses = 'Glasses: '
            if res[1] > self.glasses_threshold:
                glasses += 'True'
            else:
                glasses += 'False'
            label_res.append(glasses)

            # 帽子 (索引 0)
            hat = 'Hat: '
            if res[0] > self.threshold:
                hat += 'True'
            else:
                hat += 'False'
            label_res.append(hat)

            # 手持物 (索引 18)
            hold_obj = 'HoldObjectsInFront: '
            if res[18] > self.hold_threshold:
                hold_obj += 'True'
            else:
                hold_obj += 'False'
            label_res.append(hold_obj)

            # 包 (索引 15-17)
            bag = bag_list[np.argmax(res[15:18])]
            bag_score = res[15 + np.argmax(res[15:18])]
            bag_label = bag if bag_score > self.threshold else 'No bag'
            label_res.append(bag_label)

            # 上衣 (索引 2-7)
            upper_res = res[4:8]
            upper_label = 'Upper:'
            sleeve = 'LongSleeve' if res[3] > res[2] else 'ShortSleeve'
            upper_label += ' {}'.format(sleeve)
            for i, r in enumerate(upper_res):
                if r > self.threshold:
                    upper_label += ' {}'.format(upper_list[i])
            label_res.append(upper_label)

            # 下衣 (索引 8-13)
            lower_res = res[8:14]
            lower_label = 'Lower: '
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += ' {}'.format(lower_list[i])
                    has_lower = True
            if not has_lower:
                lower_label += ' {}'.format(lower_list[np.argmax(lower_res)])
            label_res.append(lower_label)

            # 鞋子 (索引 14)
            shoe = 'Boots' if res[14] > self.threshold else 'No boots'
            label_res.append(shoe)

            # 二值化输出
            threshold_list = [self.threshold] * len(res)
            threshold_list[1] = self.glasses_threshold
            threshold_list[18] = self.hold_threshold
            pred_res = (np.array(res) > np.array(threshold_list)).astype(np.int8).tolist()

            batch_res.append({"attributes": label_res, "output": pred_res})

        return batch_res


class FaceAttribute(object):
    """人脸属性识别后处理

    支持 40 个人脸属性

    Args:
        threshold (float): 阈值，默认 0.65
        convert_cn (bool): 是否转换为中文，默认 False
    """

    def __init__(self, threshold: float = 0.65, convert_cn: bool = False):
        self.threshold = threshold
        self.convert_cn = convert_cn

    def __call__(self, x, file_names=None):
        """执行后处理"""
        attribute_list = [
            ["CheekWhiskers", "刚长出的双颊胡须"], ["ArchedEyebrows", "柳叶眉"],
            ["Attractive", "吸引人的"], ["BagsUnderEyes", "眼袋"], ["Bald", "秃头"],
            ["Bangs", "刘海"], ["BigLips", "大嘴唇"], ["BigNose", "大鼻子"],
            ["BlackHair", "黑发"], ["BlondHair", "金发"], ["Blurry", "模糊的"],
            ["BrownHair", "棕发"], ["BushyEyebrows", "浓眉"], ["Chubby", "圆胖的"],
            ["DoubleChin", "双下巴"], ["Eyeglasses", "带眼镜"], ["Goatee", "山羊胡子"],
            ["GrayHair", "灰发或白发"], ["HeavyMakeup", "浓妆"],
            ["HighCheekbones", "高颧骨"], ["Male", "男性"],
            ["MouthSlightlyOpen", "微微张开嘴巴"], ["Mustache", "胡子"],
            ["NarrowEyes", "细长的眼睛"], ["NoBeard", "无胡子"],
            ["OvalFace", "椭圆形的脸"], ["PaleSkin", "苍白的皮肤"],
            ["PointyNose", "尖鼻子"], ["RecedingHairline", "发际线后移"],
            ["RosyCheeks", "红润的双颊"], ["Sideburns", "连鬓胡子"], ["Smiling", "微笑"],
            ["StraightHair", "直发"], ["WavyHair", "卷发"],
            ["WearingEarrings", "戴着耳环"], ["WearingHat", "戴着帽子"],
            ["WearingLipstick", "涂了唇膏"], ["WearingNecklace", "戴着项链"],
            ["WearingNecktie", "戴着领带"], ["Young", "年轻人"]
        ]
        gender_list = [["Male", "男性"], ["Female", "女性"]]
        age_list = [["Young", "年轻人"], ["Old", "老年人"]]

        batch_res = []
        index = 1 if self.convert_cn else 0

        for idx, res in enumerate(x):
            res = res.tolist() if isinstance(res, np.ndarray) else res
            label_res = []
            threshold_list = [self.threshold] * len(res)
            pred_res = (np.array(res) > np.array(threshold_list)).astype(np.int8).tolist()

            for i, value in enumerate(pred_res):
                if i == 20:
                    label_res.append(
                        gender_list[0][index] if value == 1 else gender_list[1][index]
                    )
                elif i == 39:
                    label_res.append(
                        age_list[0][index] if value == 1 else age_list[1][index]
                    )
                else:
                    if value == 1:
                        label_res.append(attribute_list[i][index])

            batch_res.append({"attributes": label_res, "output": pred_res})

        return batch_res


class VehicleAttribute(object):
    """车辆属性识别后处理

    Args:
        color_threshold (float): 颜色阈值，默认 0.5
        type_threshold (float): 类型阈值，默认 0.5
    """

    def __init__(self, color_threshold: float = 0.5, type_threshold: float = 0.5):
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"
        ]

    def __call__(self, batch_preds, file_names=None):
        """执行后处理"""
        batch_res = []
        for res in batch_preds:
            res = res.tolist() if isinstance(res, np.ndarray) else res
            label_res = []

            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])

            if res[color_idx] >= self.color_threshold:
                color_info = f"Color: ({self.color_list[color_idx]}, prob: {res[color_idx]})"
            else:
                color_info = "Color unknown"

            if res[type_idx + 10] >= self.type_threshold:
                type_info = f"Type: ({self.type_list[type_idx]}, prob: {res[type_idx + 10]})"
            else:
                type_info = "Type unknown"

            label_res = f"{color_info}, {type_info}"

            threshold_list = [self.color_threshold] * 10 + [self.type_threshold] * 9
            pred_res = (np.array(res) > np.array(threshold_list)).astype(np.int8).tolist()

            batch_res.append({"attributes": label_res, "output": pred_res})

        return batch_res


class TableAttribute(object):
    """表格属性识别后处理

    Args:
        source_threshold (float): 来源阈值
        number_threshold (float): 数量阈值
        color_threshold (float): 颜色阈值
        clarity_threshold (float): 清晰度阈值
        obstruction_threshold (float): 遮挡阈值
        angle_threshold (float): 角度阈值
    """

    def __init__(
        self,
        source_threshold: float = 0.5,
        number_threshold: float = 0.5,
        color_threshold: float = 0.5,
        clarity_threshold: float = 0.5,
        obstruction_threshold: float = 0.5,
        angle_threshold: float = 0.5,
    ):
        self.source_threshold = source_threshold
        self.number_threshold = number_threshold
        self.color_threshold = color_threshold
        self.clarity_threshold = clarity_threshold
        self.obstruction_threshold = obstruction_threshold
        self.angle_threshold = angle_threshold

    def __call__(self, batch_preds, file_names=None):
        """执行后处理"""
        batch_res = []

        for res in batch_preds:
            res = res.tolist() if isinstance(res, np.ndarray) else res
            label_res = []

            source = 'Scanned' if res[0] > self.source_threshold else 'Photo'
            number = 'Little' if res[1] > self.number_threshold else 'Numerous'
            color = 'Black-and-White' if res[2] > self.color_threshold else 'Multicolor'
            clarity = 'Clear' if res[3] > self.clarity_threshold else 'Blurry'
            obstruction = (
                'Without-Obstacles' if res[4] > self.obstruction_threshold
                else 'With-Obstacles'
            )
            angle = 'Horizontal' if res[5] > self.angle_threshold else 'Tilted'

            label_res = [source, number, color, clarity, obstruction, angle]

            threshold_list = [
                self.source_threshold, self.number_threshold,
                self.color_threshold, self.clarity_threshold,
                self.obstruction_threshold, self.angle_threshold
            ]
            pred_res = (np.array(res) > np.array(threshold_list)).astype(np.int8).tolist()

            batch_res.append({"attributes": label_res, "output": pred_res})

        return batch_res

