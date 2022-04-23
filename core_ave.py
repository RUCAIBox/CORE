import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender


class COREave(SequentialRecommender):
    def __init__(self, config, dataset):
        super(COREave, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']

        self.sess_dropout = nn.Dropout(config['sess_dropout'])
        self.item_dropout = nn.Dropout(config['item_dropout'])
        self.temperature = config['temperature']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def ave_net(self, item_seq):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        # Representation-Consistent Encoder (RCE)
        alpha = self.ave_net(item_seq)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight
        # no dropout for evaluation
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores
