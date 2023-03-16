"""
@File  : skp_json_parser.py
@Author: tao.jing
@Date  : 2022/4/5
@Desc  :
"""
import os
import json


__all__ = [
    'SKPJsonParser'
]


class SKPJsonParser(object):
    def __init__(self, annot_path, load_ratio=1.0):
        self.annot_path = annot_path
        self.load_ratio = load_ratio

        assert os.path.isfile(self.annot_path), \
            f'Annotation json not exist: {self.annot_path}'

        self.parse_annot_json()

    def parse_annot_json(self):
        self.skp_db = None
        with open(self.annot_path, 'r') as f:
            self.skp_db = json.load(f)['smoke_kp']

        assert isinstance(self.skp_db, list) and len(self.skp_db) > 0, \
            f'Invalid annot file.'
        assert 0.0 < self.load_ratio <= 1.0, \
            f'[SKPJsonParser] Invalid load ratio {self.load_ratio}'

        self.skp_db = self.skp_db[:int(len(self.skp_db) * self.load_ratio)]

    def __len__(self):
        return len(self.skp_db)

    def __getitem__(self, index):
        assert index < len(self.skp_db), \
            f'Invalid index of self.skp_db, index: {index}, len: {len(self.skp_db)}'
        return self.skp_db[index]



if __name__ == '__main__':
    annot_path = 'D:\\Projects\\key_point\\kp_paddle\\local_data\\datalist\\smoke_keypoint.json'
    SKPJsonParser(annot_path)