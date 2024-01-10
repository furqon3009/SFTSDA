import sys

sys.path.append('../ADATIME')

import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class TargetTest(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(TargetTest, self).__init__(args)

        self.last_results = None
        self.best_results = None
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        self.run_description)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        non_adapted = checkpoint['non_adapted']
        adapted_last = checkpoint['last']
        adapted_best = checkpoint['best']
        return non_adapted, adapted_last, adapted_best

    def build_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        return algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device).to(self.device)

    # def scenario_test(self):
    #
    #     results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
    #     last_results = pd.DataFrame(columns=results_columns)
    #     best_results = pd.DataFrame(columns=results_columns)
    #
    #     # Trainer
    #     for src_id, trg_id in self.dataset_configs.scenarios:
    #         for run_id in range(self.num_runs):
    #             # fixing random seed
    #             fix_randomness(run_id)
    #
    #             # Logging
    #             self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))
    #
    #             self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
    #
    #             # Load data
    #             self.load_data(src_id, trg_id)
    #
    #             # Build model
    #             self.algorithm = self.build_model()
    #
    #             # Load chechpoint
    #             source_only_model, adapted_last, adapted_best = self.load_checkpoint(self.scenario_log_dir)
    #
    #             # Testing the model
    #             non_adapted_metrics = self.model_test(source_only_model)
    #             last_metrics = self.model_test(adapted_last)
    #             best_metrics = self.model_test(adapted_best)
    #
    #             # Append results to tables
    #             src_only_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,non_adapted_metrics)
    #             last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,last_metrics)
    #             best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id, best_metrics)
    #
    #     summary_src_only = {metric: np.mean(src_only_results[metric]) for metric in results_columns[2:]}
    #     summary_last = {metric: np.mean(last_results[metric]) for metric in results_columns[2:]}
    #     summary_best = {metric: np.mean(best_results[metric]) for metric in results_columns[2:]}
    #
    #     # Calculate and append mean and std to tables
    #     last_results = self.add_mean_std_table(last_results, results_columns)
    #     best_results = self.add_mean_std_table(best_results, results_columns)
    #
    #     # Save tables to file if needed
    #     self.save_tables_to_file(last_results, 'last_results')
    #     self.save_tables_to_file(best_results, 'best_results')
    #
    #     for summary_name, summary in [('src_only', summary_src_only), ('Last', summary_last), ('Best', summary_best)]:
    #         for key, val in summary.items():
    #             print(f'{summary_name}: {key}\t: {val:2.4f}')

    def scenario_test(self):
        # Define columns for results tables
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]

        # Initialize results tables
        last_results = pd.DataFrame(columns=results_columns)
        best_results = pd.DataFrame(columns=results_columns)
        non_adapted_results = pd.DataFrame(columns=results_columns)

        # Train and test models for each scenario and run
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # Fix random seed and set up logging directory
                fix_randomness(run_id)
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                # Load data, build model, and load checkpoint
                self.load_data(src_id, trg_id)
                self.algorithm = self.build_model()
                non_adapted_last, adapted_last, adapted_best = self.load_checkpoint(self.scenario_log_dir)

                # Test models and append results to tables
                non_adapted_metrics = self.model_test(non_adapted_last)
                last_metrics = self.model_test(adapted_last)
                best_metrics = self.model_test(adapted_best)
                non_adapted_results = self.append_results_to_tables(non_adapted_results, f"{src_id}_to_{trg_id}", run_id, non_adapted_metrics)
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id, last_metrics)
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id, best_metrics)

        # Calculate mean and std of each metric for each scenario and print results
        summary_src_only = non_adapted_results[results_columns[2:]].mean()
        summary_last = last_results[results_columns[2:]].mean()
        summary_best = best_results[results_columns[2:]].mean()
        for summary_name, summary in [('src_only', summary_src_only), ('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.iteritems():
                print(f'{summary_name}: {key}\t: {val:2.4f}')

        # Add mean and std tables to results tables and save to file
        last_results = self.add_mean_std_table(last_results, results_columns)
        best_results = self.add_mean_std_table(best_results, results_columns)
        self.save_tables_to_file(last_results, 'last_results')
        self.save_tables_to_file(best_results, 'best_results')

    def model_test(self, chkpoint):
        # Load the model dictionary
        self.algorithm.network.load_state_dict(chkpoint)

        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _ in self.trg_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, _ = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat(preds_list)
        self.full_labels = torch.cat(labels_list)

        return self.calculate_metrics()


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='EVD', type=str,
                        help='')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../../data', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()

    tester = TargetTest(args)
    tester.scenario_test()
