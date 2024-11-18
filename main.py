from types import SimpleNamespace
import torch
from torch import nn
import numpy as np
import logging
import math
import random
import csv
from device_env import Evaluator, base_path
import os
import sys
import pickle
import warnings
from utils.logger import info, error
from autos.train import autos_selection

warnings.filterwarnings('ignore')

from plato.servers import fedavg
from plato.config import Config
from plato.clients import simple
from plato.trainers import basic



class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def process_loss(self, outputs, labels) -> torch.Tensor:
        """Returns the loss from CrossEntropyLoss, and records the sum of
        squaures over per_sample loss values."""
        loss_func = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_func(outputs, labels)

        # Stores the sum of squares over per_sample loss values
        self.run_history.update_metric(
            "train_squared_loss_step",
            sum(np.power(per_sample_loss.cpu().detach().numpy(), 2)),
        )

        return torch.mean(per_sample_loss)

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return self.process_loss
    
class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        train_squared_loss_step = self.trainer.run_history.get_metric_values(
            "train_squared_loss_step"
        )

        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * sum(train_squared_loss_step)
        )

        # power_config = Config().clients.power_config
        delta = 5  
        # power_init = self.client_id//len(power_config) + random.uniform(-delta, delta)

        # report.client_power = power_init

        return report

class Server(fedavg.Server):
    """A federated learning server using oort client selection."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # Clients that will no longer be selected for future rounds.
        self.blacklist = []

        # All clients' utilities
        self.client_utilities = {}

        # All clients‘ training times
        self.client_durations = {}

        # Keep track of each client's last participated round.
        self.client_last_rounds = {}

        # Number of times that each client has been selected
        self.client_selected_times = {}

        # The desired duration for each communication round
        self.desired_duration = Config().server.desired_duration

        self.explored_clients = []
        self.unexplored_clients = []

        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration

        self.penalty = Config().server.penalty

        self.penalty_beta=Config().server.penalty_beta

        self.total_clients = Config().clients.total_clients

        # Keep track of statistical utility history.
        self.util_history = []

        # Cut off for sampling client utilities
        self.cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )

        # Times a client is selected before being blacklisted
        self.blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )


        self.selction_model_state_dict = None

    def configure(self) -> None:
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_durations = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_power = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        self.round_energy = 0 

        self.unexplored_clients = list(range(1, self.total_clients + 1))

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        for update in updates:
            # Extract statistical utility and local training times
            self.client_utilities[update.client_id] = update.report.statistical_utility
            self.client_durations[update.client_id] = update.report.training_time
            self.client_last_rounds[update.client_id] = self.current_round
            # self.client_power[update.client_id] = update.report.client_power

            # Calculate client utilities of explored clients --- oort
            self.client_utilities[update.client_id] = self.calc_client_util(
                update.client_id
            )

        # Adjust pacer
        self.util_history.append(
            sum(update.report.statistical_utility for update in updates)
        )

        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window : -self.step_window]
            )
            current_pacer_rounds = sum(self.util_history[-self.step_window :])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step

        # Blacklist clients who have been selected self.blacklist_num times
        for update in updates:
            if self.client_selected_times[update.client_id] > self.blacklist_num:
                self.blacklist.append(update.client_id)

    def choose_clients_fovar(self, clients_pool, clients_count):   
           """Choose a subset of the clients to participate in each round."""
           selected_clients = []
          
           return selected_clients
    
    def choose_clients_fedmarl(self, clients_pool, clients_count):   
           """Choose a subset of the clients to participate in each round."""
           selected_clients = []
          
           return selected_clients

    def choose_clients_oort(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []
	#
        return selected_clients

    def convert_to_binary_list(self, selected_indices, list_size):
        binary_list = [0] * list_size
        for index in selected_indices:
            if 0 <= index < list_size:
                binary_list[index] = 1
        return torch.FloatTensor(binary_list)

    def gen_device_selection(self, device_eval_,clients_pool, clients_count):   
        # max_accuracy, optimal_set, k = gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=300)
        for i in range(100):
            selected_numbers = self.choose_clients_oort(clients_pool, clients_count) 
            performances = self.cal_selected_utility(selected_numbers)
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))   
            device_eval_._store_history(selected_numbers, performances)

            selected_numbers = self.choose_clients_fedmarl(clients_pool, clients_count) 
            performances = self.cal_selected_utility(selected_numbers)
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))   
            device_eval_._store_history(selected_numbers, performances)
        
            selected_numbers = self.choose_clients_favor(clients_pool, clients_count) 
            performances = self.cal_selected_utility(selected_numbers)
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))   
            device_eval_._store_history(selected_numbers, performances)
        


    def choose_clients(self, clients_pool, clients_count):
        # best_selection_test = self.choose_clients_oort(clients_pool, clients_count) 
        if self.client_utilities == self.client_durations:
            best_selection_test = self.choose_clients_oort(clients_pool, clients_count) 
        else:
           
            device_eval = Evaluator()
            self.gen_device_selection(device_eval,clients_pool, clients_count)
            file_path = f"{base_path}/history/device_env.pkl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(f'{base_path}/history/device_env.pkl', 'wb') as f: 
                pickle.dump(device_eval, f)
            
           
            new_selection, self.selction_model_state_dict = autos_selection(self.total_clients,self.selction_model_state_dict)
            best_selection_test = None
            best_optimal_test = -1000
            for s in new_selection:
                indice_select = torch.arange(0,self.total_clients)[s.operation == 1] 
                test_result = self.cal_selected_utility(indice_select.tolist())
                if test_result > best_optimal_test:
                    best_selection_test = indice_select.tolist()
                    best_optimal_test = test_result
                    info(f'found best on test : {best_optimal_test}')
            for client in best_selection_test:
                self.client_selected_times[client] += 1
            # for client_id in best_selection_test:
            #     self.unexplored_clients.remove(client_id)


        info(f'found test generation in our method! the choice is {best_selection_test}')

        self.clients_per_round =len(best_selection_test)
        return best_selection_test


    def calc_client_util(self, client_id):
        """Calculate the client utility."""
        client_utility = self.client_utilities[client_id] + math.sqrt(
            0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
        )

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (     
                self.desired_duration / self.client_durations[client_id]
            ) ** self.penalty
            client_utility *= global_utility
        return client_utility


    def cal_selected_utility(self, selected_clients):
        client_utility=[]
        # client_energy =[]
        for client_id in selected_clients:
            # client_energy.append(self.client_durations[client_id]*self.client_power[client_id])
            # desired_energy = self.desired_duration*self.client_power[client_id]
            # if desired_energy < self.client_durations[client_id]*self.client_power[client_id]:
            #     energy_utility = (
            #         self.desired_energy[client_id]/ (self.client_durations[client_id]*self.client_power[client_id])
            #     )** self.penalty_beta
            client_utility.append(self.client_utilities[client_id] )

        # self.round_energy = float(sum(client_energy))
        total_utility = float(sum(client_utility)) 
        return total_utility
    
    # def get_logged_items(self) -> dict:
    #     super().get_logged_items() 
    #     logged_items["energy"] = self.round_energy  


def main():
    np.random.seed(1)
    trainer = Trainer
    client = Client(trainer=trainer)
    server = Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()