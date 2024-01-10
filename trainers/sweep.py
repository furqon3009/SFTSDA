import sys

sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import wandb

import collections
import argparse
import warnings
import sklearn.exceptions

from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs

from utils import AverageMeter

from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)

    def train(self):
        run = wandb.init(config=self.hparams, dir=self.home_path)

        # create tables for results and risks
        columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = wandb.Table(columns=columns, allow_mixed_types=True)
        columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = wandb.Table(columns=columns, allow_mixed_types=True)

        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # set random seed and create logger
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Train model
                train_models = self.train_model()

                # calculate metrics and risks
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results.add_data(scenario, run_id, *metrics)
                table_risks.add_data(scenario, run_id, *risks)

        # calculate overall metrics and risks
        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        # finish the run
        run.finish()


if __name__ == "__main__":
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='MAPU', type=str,
                        help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../../data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ======== sweep settings =====================
    parser.add_argument('--num_sweeps', default=1, type=str, help='Number of sweep runs')

    # We run sweeps using wandb plateform, so next parameters are for wandb.
    parser.add_argument('--sweep_project_wandb', default='MAPU_test', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str,  help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--hp_search_strategy', default="random", type=str, help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_minimize', default="src_risk", type=str,  help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs/sweep_logs', type=str,
                        help='Directory containing all experiments')

    args = parser.parse_args()

    trainer = Trainer(args)

    trainer.train()
