import numpy as np

import torch
from typing import Dict

from labml import lab, monit
from labml.utils import download


class CoraDataset:
    '''
    # Cora Dataset

    2,708건의 Scientific Publication을 포함하는 데이터셋.
    7개의 class로 분류되며, 각각의 publication을 노드로 하여 인용 네트워크를 이룬다.
    해당 인용 네트워크는 5,429개의 link(edge)를 가지는 Directed Graph이다.

    label prediction과 link prediction 등에서 Benchmarks Dataset로 사용된다.
    '''
    labels: torch.Tensor
    classes: Dict[str, int]
    features: torch.Tensor
    adj_mat: torch.Tensor

    # Download the Dataset
    @staticmethod
    def _download():
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() /
                                 'cora.tgz', lab.get_data_path())

    def __init__(self, include_edges: bool = True):
        self.include_edges = include_edges
        self._download()

        with monit.section('Read Content'):
            content = np.genfromtxt(
                str(lab.get_data_path() / 'cora/cora.content'), dtype=np.dtype(str))

        with monit.section('Read Citations'):
            citations = np.genfromtxt(
                str(lab.get_data_path() / 'cora/cora.cites'), dtype=np.int32)

        # Get the feature vectors
        features = torch.tensor(np.array(content[:, 1:-1], dtype=np.float32))

        # Normalize
        self.features = features / features.sum(dim=1, keepdim=True)

        # Get class names and assign integer tags
        self.classes = {s: i for i, s in enumerate(set(content[:, -1]))}
        # to labels
        self.labels = torch.tensor([self.classes[i]
                                   for i in content[:, -1]], dtype=torch.long)

        # Get paper ids
        paper_ids = np.array(content[:, 0], dtype=np.int32)
        # Map id to idx
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}

        # Empty adj_mat
        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)

        # Mark the citations in adj_mat
        if self.include_edges:
            for e in citations:
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]

                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True
