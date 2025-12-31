import numpy as np
import torch
import math
import torch.nn.functional as f
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch

def patched_run_epoch(self, num_epoch):
    num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)
    self._model.train()
    epoch_loss = 0

    for _ in range(num_batches_per_epoch):
        self._optimizer.zero_grad()
        u_idx = np.random.choice(self.warm_user_ids, size=self.batch_size)
        user_batch_tensor = self.URM_train[u_idx]

        user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                    user_batch_tensor.indices,
                                                    user_batch_tensor.data,
                                                    size=user_batch_tensor.shape, 
                                                    dtype=torch.float32, 
                                                    device=self.device).to_dense()

        logits, KL, mu_q, std_q, epsilon, sampled_z = self._model.forward(user_batch_tensor)
        log_softmax_var = f.log_softmax(logits, dim=1)
        neg_ll = - torch.mean(torch.sum(log_softmax_var * user_batch_tensor, dim=1))
        l2_reg = self._model.get_l2_reg()
        anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps) if self.total_anneal_steps > 0 else self.anneal_cap

        loss = neg_ll + anneal * KL + l2_reg * self.l2_reg
        self.update_count += 1
        loss.backward()
        epoch_loss += loss.item()
        self._optimizer.step()

    self._print("Loss {:.2E}".format(epoch_loss))
    self._model.eval()

def patched_compute_item_score(self, user_id_array, items_to_compute = None):
    user_batch_tensor = self.URM_train[user_id_array]
    user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                user_batch_tensor.indices,
                                                user_batch_tensor.data,
                                                size=user_batch_tensor.shape, 
                                                dtype=torch.float32,
                                                device=self.device).to_dense()

    with torch.no_grad():
        self._model.eval()
        logits, _, _, _, _, _ = self._model.forward(user_batch_tensor)

    item_scores_to_compute = logits.cpu().detach().numpy()
    if items_to_compute is not None:
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
        item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
    else:
        item_scores = item_scores_to_compute
    return item_scores

# Applicazione della patch
MultVAERecommender_PyTorch._run_epoch = patched_run_epoch
MultVAERecommender_PyTorch._compute_item_score = patched_compute_item_score