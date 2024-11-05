from mmdet.models.builder import LOSSES
import pdb
import torch

@LOSSES.register_module()
class Hierarchical_Task_Learning:
    def __init__(self,epoch_iters,
                      slide_window_epochs,
                      total_epochs,
                      loss_graph):
        #self.index2term = [*epoch0_loss.keys()]
        #self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        #self.stat_epoch_nums = stat_epoch_nums
        #self.past_losses=[]
        self.slide_window_epochs = slide_window_epochs
        self.epoch_iters = epoch_iters
        self.total_epochs = total_epochs
        self.loss_graph = loss_graph 
                  
        self.intra_epoch_losses = []
        self.inter_epoch_losses = []
        self.current_epochs = 0
    def update_losses(self,current_loss):
        self.intra_epoch_losses.append([current_loss[key].mean().detach().cpu() for key in current_loss.keys()])
        if len(self.intra_epoch_losses)==self.epoch_iters:
            self.inter_epoch_losses.append(torch.tensor(self.intra_epoch_losses).mean(0))
            self.intra_epoch_losses = []
            self.current_epochs += 1
        if len(self.inter_epoch_losses) > self.slide_window_epochs:
            self.inter_epoch_losses.pop(0)
    def compute_weights(self,current_loss):
        loss_weights = {}
        if self.current_epochs == 0:  #initialization (epoch0_loss)
            for term in self.loss_graph:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        elif self.current_epochs > 0:
            for term in self.loss_graph:
                if len(self.loss_graph[term])==0:
                    loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
                else:
                    loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
            if self.current_epochs >= self.slide_window_epochs:
                try:
                    past_loss = torch.stack(self.inter_epoch_losses)
                except:
                    pdb.set_trace()
                mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
                if not hasattr(self, 'init_diff'):
                    self.init_diff = mean_diff
                    index2term = [*current_loss.keys()]
                    self.term2index = {term:index2term.index(term) for term in index2term} 
                c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)

                time_value = min(((self.current_epochs-self.slide_window_epochs)/(self.total_epochs-self.slide_window_epochs)),1.0)
                for current_topic in self.loss_graph:
                    if len(self.loss_graph[current_topic])!=0:
                        control_weight = 1.0
                        for pre_topic in self.loss_graph[current_topic]:
                            control_weight *= c_weights[0][self.term2index[pre_topic]]      
                        loss_weights[current_topic] = time_value**(1-control_weight)
        for key in current_loss.keys():
            current_loss[key] *= loss_weights[key] 