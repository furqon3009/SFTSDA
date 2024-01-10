import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking, get_distances, soft_k_nearest_neighbors, refine_predictions, eval_and_label_dataset, update_labels
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl, contrastive_loss, TEntropyLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from algorithms.moco import *
import math


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (_, src_y, _, _, _, src_x) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (_, _, trg_idx, _, _, trg_x) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


class SFTSDA(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SFTSDA, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.momentum_feature_extractor = backbone(configs)
        self.momentum_classifier = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.momentum_network = nn.Sequential(self.momentum_feature_extractor, self.momentum_classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.momentum_network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'masking_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, trg_test_dataloader, avg_meter, logger, num_neighbors, configs, temp_length, trg_train_dataset_length):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()
        momentum_net = self.network.state_dict()

        moco_model = AdaMoCo(src_model = self.network, momentum_model = self.momentum_network, features_length=configs.final_out_channels, num_classes=self.configs.num_classes, dataset_length=trg_train_dataset_length, temporal_length=temp_length)
        banks = eval_and_label_dataset(0, self.feature_extractor, self.classifier, None, trg_test_dataloader, trg_dataloader, num_neighbors)
        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        N = 12
        alpha = 0.005
        betha = 1e-4

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            self.feature_extractor.train()
            self.classifier.train()
            self.network.train()
            moco_model.train()

            for step, (trg_x, y, trg_idx, src_x_strong, src_x_strong2, x) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                src_x_strong, src_x_strong2 = src_x_strong.float().to(self.device), src_x_strong2.float().to(self.device)

                self.optimizer.zero_grad()

                trg_q, trg_k = (
                    trg_x[0].to("cuda"),
                    trg_x[1].to("cuda"),
                )

                outputs_tcrs = []
                with torch.no_grad():
                    for jj in range(N-2):
                        outputs_tcrs.append(moco_model(trg_x, self.feature_extractor, self.classifier, self.momentum_feature_extractor, self.momentum_classifier, cls_only=True)[1]) 
                    outputs_tcr = torch.stack(outputs_tcrs).mean(0)
                    logits_w = torch.nn.functional.softmax(outputs_tcr, dim=1)
                    probs_w, pseudo_labels_w = logits_w.max(dim=1)
                    
                trg_feat, trg_pred  = moco_model(trg_x, self.feature_extractor, self.classifier, self.momentum_feature_extractor, self.momentum_classifier, tcr_only=True)
                
                # select evidential vs softmax probabilities
                trg_prob = torch.nn.Softmax(dim=1)(trg_pred)

                # pseudo label refinement
                with torch.no_grad():
                    probs = trg_prob
                    pseudo_labels, probs_refine, _, _ = refine_predictions(trg_feat, probs, banks, num_neighbors)

                # Sample Selection 
                pred_batch, _ = trg_prob.max(dim=1)
                pred_start = torch.nn.functional.softmax(torch.squeeze(torch.stack(outputs_emas)), dim=2).max(2)[0] 
                
                ## Confidence Based Selection
                pred_con = pred_start
                conf_thres = pred_con.mean()
                confidence_sel = pred_con.mean(0) > conf_thres
                conf_th = pred_con.mean()

                ## Uncertainty Based Selection
                pred_std = pred_start.std(0)
                uncertainty_thres = pred_std.mean(0)
                uncertainty_sel = pred_std < uncertainty_thres
                uncer_th = pred_std.mean(0)

                ## Confidence and Uncertainty Based Selection
                truth_array = torch.logical_and(uncertainty_sel, confidence_sel)
                ind_keep = torch.squeeze(truth_array.nonzero(), dim=-1)
                ind_remove = torch.squeeze((~truth_array).nonzero(), dim=-1)

                try:
                    ind_total = torch.cat((ind_keep, ind_remove), dim=0)
                except:
                    ind_total = ind_remove

                ## Confidence Score Difference (DoC) Based Selection
                if ind_remove.numel():
                    threshold = torch.zeros(len(ind_remove))
                    num = 0
                    for kk in ind_remove:
                        out = torch.squeeze(outputs_tcr[kk])
                        out , _ = out.sort(descending=True)
                        threshold[num] = out[0] - out[1]
                        num += 1

                    pre_threshold = threshold.mean(0) 
                    truth_array1 = threshold > pre_threshold
                    truth_array2 = pred_std[ind_remove] < pred_std[ind_remove].mean(0)                   ## Add Underconfident Clean Samples 
                    truth_array = torch.logical_and(truth_array1.cuda(), truth_array2.cuda())
                    ind_add = truth_array.nonzero()
                    
                    try:
                        ind_keep = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove[ind_add])), dim=0)
                        ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
                    except:
                        pass

                try:
                    ### Apply Class-Balancing (Only the selected Samples) ###
                    unique_labels, counts = pseudo_labels_w[ind_keep].unique(return_counts = True)
                    min_count = torch.min(counts)

                    ## For Missing Classes
                    if len(counts) < self.configs.num_classes:
                        counts_new = torch.ones(self.configs.num_classes)
                        missing_classes = [ii for ii in range(self.configs.num_classes) if ii not in unique_labels]
                        
                        for kk in missing_classes:
                            indices = (pseudo_labels_w == kk).nonzero(as_tuple=True)[0] # find missing classes in pseudo labels
                            
                            if indices.numel()>0 and ind_keep.numel()>0:
                                probs2,_  = probs_w[indices]
                                _ , index_miss = probs2.sort(descending=True)                              
                    
                                try:
                                    ind_keep  = torch.cat((ind_keep, indices[index_miss[0:min_count]]))
                                    ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
                                except:
                                    pass
                                
                                counts_new[kk] = 1 
                            else:
                                counts_new[kk] = 1
                        
                        ## Other Classes
                        num = 0
                        for nn in unique_labels:
                            counts_new[nn] = counts[num]
                            num += 1
                    else:
                        counts_new = counts 

                    trg_cls_loss = self.cross_entropy(trg_pred[ind_keep], pseudo_labels_w[ind_keep])

                except:
                    trg_cls_loss = torch.mean((torch.squeeze(outputs_tcr)-torch.squeeze(trg_pred))**2)

                
                 ### Propagation Loss ###
                ## If the clean selected set is empty, calculate loss for all samples  
                try:
                    propagation_loss = torch.mean((torch.squeeze(outputs_tcr[ind_remove])-torch.squeeze(trg_pred[ind_remove]))**2)
                except:
                    propagation_loss = 0

                # contrastive learning
                _, logits_q, logits_ctr, keys = moco_model(src_x_strong, self.feature_extractor, self.classifier, self.momentum_feature_extractor, self.momentum_classifier, src_x_strong2)
                loss_ctr = contrastive_loss(logits_ins=logits_ctr, pseudo_labels=moco_model.mem_labels[trg_idx], mem_labels=moco_model.mem_labels[moco_model.idxs])
                moco_model.update_memory(epoch, trg_idx.to(self.device), keys, pseudo_labels, y.to(self.device))

                # uncertainty reduction loss
                loss_ur = TEntropyLoss(t=configs.temp)(trg_prob)
                
                # curriculum learning 
                d = uncertainty_thres/conf_thres
                if epoch == 1:
                    mu_r = 1
                    mu_c = 0.5
                    mu_u = 0.5
                else:
                    mu_r = mu_r * (1-(alpha*math.exp(-1/d)))
                    mu_c = mu_c * math.exp(-betha)
                    mu_u = mu_u * math.exp(-betha)

                '''
                Overall objective loss
                '''
                loss = mu_r*trg_cls_loss + (1-mu_r)*propagation_loss + mu_c*loss_ctr + mu_u*loss_ur

                update_labels(banks, trg_idx, trg_feat, trg_pred)
                loss.backward()
                self.optimizer.step()
            
                losses = {'target_cls_loss': trg_cls_loss.detach().item(),'contr_loss': loss_ctr.detach().item(), 'prop_loss': propagation_loss.detach().item(), 'ur_loss': loss_ur.detach().item(),  'Total_loss': loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())
            
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

