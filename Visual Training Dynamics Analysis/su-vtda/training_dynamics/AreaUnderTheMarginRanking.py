import torch

class AreaUnderTheMarginRanking():
    def __init__(self):
        # hist_delta_AUM_current_epoch dimensions: [n_sample, 2 (from in_logit & max(out_logits))]
        self.hist_delta_AUM_current_epoch = torch.zeros(size=(0, 2))
        # hist_delta_AUM dimensions: [n_epoch, n_sample, in_logit & max(out_logits)]
        self.hist_delta_AUM = torch.zeros(size=(0, 0, 2))
        self.reference_sample_idx = []

    def accumulate(self, batch_logits, batch_ids, batch_targets):
        """ To be called after batch prediction"""
        for img_logit, img_id, img_target in zip(batch_logits.split([1] * len(batch_logits), dim=0),
                                                 batch_ids.split([1] * len(batch_logits), dim=0),
                                                 batch_targets.split([1] * len(batch_logits), dim=0)):
            img_logit = img_logit.squeeze(dim=0)
            target_logit = img_logit[img_target]
            if img_target < len(img_logit) - 1:
                notarget_logits = torch.cat([img_logit[:img_target], img_logit[img_target + 1:]], dim=0)
            else:
                notarget_logits = img_logit[:img_target]
            notarget_logits = notarget_logits.max()
            self.hist_delta_AUM_current_epoch = torch.cat(
                [self.hist_delta_AUM_current_epoch, torch.tensor([[target_logit, notarget_logits]])], dim=0)

    def accumulate_epoch(self):
        """ To be called at the end of each epoch"""
        if len(self.hist_delta_AUM) == 0:
            self.hist_delta_AUM = self.hist_delta_AUM_current_epoch.unsqueeze(dim=0)
        else:
            self.hist_delta_AUM = torch.cat([self.hist_delta_AUM, self.hist_delta_AUM_current_epoch.unsqueeze(dim=0)],
                                            dim=0)
        self.hist_delta_AUM_current_epoch = torch.zeros(size=(0, 2))