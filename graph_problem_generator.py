import os, sys
import numpy as np
import networkx as nx
import torch
from utils import flat
import random
from metro_generate_data import *


PHASES = ['State', 'Goal', 'Plan', 'Solve']

phase_to_ix = {word: i for i, word in enumerate(PHASES)}
ix_to_phase = {i: word for i, word in enumerate(PHASES)}


class ShortestPathGraphData():
    def __init__(self, num_nodes=10, batch_size=1, plan_phase=3, min_path_len=2, max_path_len=2, cuda=False):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.plan_len = plan_phase
        self.min_path_len = min_path_len
        self.max_path_len = max_path_len

        self.STATE, self.INIT_STATE = '', ''
        self.current_index, self.cuda = 0, cuda
        self.goal = None
        self.shortest_path = None
        self.current_edges, self.current_graph = None, None

        self.masks = []
        self.make_new_graph()

        self.phase_oh = torch.eye(len(PHASES)).float()
        self.blnk_vec = torch.zeros(1, 61).float()
    

    @property
    def nn_in_size(self):
        return self.blnk_vec.size(-1) + self.phase_oh.size(-1)
    

    @property
    def nn_out_size(self):
        return self.blnk_vec.size(-1)
    

    def make_new_graph(self):
        nx_graph = None
        self.shortest_path = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        while len(self.shortest_path)-1 > self.max_path_len or len(self.shortest_path)-1 < self.min_path_len:
            flag = random.random() < 0.5
            if flag:
                alpha = random.random()
                nx_graph = nx.watts_strogatz_graph(self.num_nodes, 2, alpha)
            else:
                nx_graph = nx.barabasi_albert_graph(self.num_nodes, 1)
            self.current_graph = nx.to_dict_of_lists(nx_graph)
            self.current_edges = self.nx_edges_to_vecs(nx.to_edgelist(nx_graph))
            start_id, goal_id = random.sample(range(self.num_nodes), 2)
            self.INIT_STATE, self.goal = int(start_id), int(goal_id)
            self.shortest_path = bfs_shortest_path(self.current_graph, self.INIT_STATE, self.goal)
        self.STATE = self.INIT_STATE
        state = torch.zeros(len(self.current_edges))
        goal = torch.ones(1)
        plan = torch.ones(self.plan_len) * 2
        resp = torch.ones(self.max_path_len) * 3
        self.masks = torch.cat([state, goal, plan, resp], 0).long()
        self.current_index = 0
        return self.masks
    

    def nx_edges_to_vecs(self, nx_edges):
        out_vecs = []
        for _t in nx_edges:
            out_vecs.append([_t[0], _t[1]])
            out_vecs.append([_t[1], _t[0]])
        return out_vecs


    def vecs_to_ixs(self, vecs):
        out_idxs = []
        for vec in vecs:
            out_idxs.append(self.vec_to_ix(vec))
        return out_idxs


    def vec_to_ix(self, vec):
        out_ix = []
        from_e = "{0:03d}".format(vec[0])
        to_e = "{0:03d}".format(vec[1])
        for _s in from_e:
            one_hot = np.eye(10)[[int(_s)]].tolist()[0]
            out_ix.extend(one_hot)
        out_ix.append(0.0)
        for _s in to_e:
            one_hot = np.eye(10)[[int(_s)]].tolist()[0]
            out_ix.extend(one_hot)
        return out_ix


    def ix_to_vec(self, ix):
        out_vec = []
        start_idx = 0
        from_str = ""
        to_str = ""
        for i in range(6):
            tmp_arr = np.array(ix[i*10 + start_idx : (i+1)*10 + start_idx])
            tmp_val = np.argmax(tmp_arr)
            if i < 3:
                from_str = from_str + str(tmp_val)
            else:
                to_str = to_str + str(tmp_val)
            if i == 2:
                start_idx = 1
        out_vec = [int(from_str), int(to_str)]
        return out_vec
    

    def minimums(self, some_dict):
        positions = [] # output variable
        min_value = float("inf")
        for k, v in some_dict.items():
            if v == min_value:
                positions.append(k)
            if v < min_value:
                min_value = v
                positions = [] # output variable
                positions.append(k)
        return positions
    

    def get_actions(self):
        next_nodes = self.current_graph[self.STATE]
        best_nodes = None
        if len(next_nodes) > 1:
            best_nodes_can = {}
            for _node in next_nodes:
                _spath = bfs_shortest_path(self.current_graph, _node, self.goal)
                best_nodes_can[_node] = len(_spath)
            best_nodes = self.minimums(best_nodes_can)
        else:
            best_nodes = next_nodes
        return best_nodes, next_nodes
    

    def update_state(self, action):
        self.STATE = action
    

    def strip_ix_mask(self, ix_input_vec):
        phase_size = len(PHASES)
        ixs_vec = ix_input_vec[:, phase_size:]
        phase_vec = ix_input_vec[:, :phase_size]
        return phase_vec, ixs_vec
    

    def ix_input_to_ixs(self, ix_input_vec, grouping=None):
        phase_size = self.phase_oh.size(-1)
        ixs_vec = ix_input_vec[:, phase_size:]
        output_vec = [
            ixs_vec[:, :10],
            ixs_vec[:, 10:20],
            ixs_vec[:, 20:30],
            ixs_vec[:, 31:41],
            ixs_vec[:, 41:51],
            ixs_vec[:, 51:61]
        ]
        return output_vec
    

    def getitem(self, batch=1):
        if self.current_index >= len(self.masks):
            self.make_new_graph()
        
        phase = self.masks[self.current_index]
        if phase == 0:
            inputs = np.array(self.vec_to_ix(self.current_edges[self.current_index])).reshape((1, 61))
            inputs = torch.from_numpy(inputs)
        elif phase == 1:
            inputs = np.array(self.vec_to_ix([self.INIT_STATE, self.goal])).reshape((1, 61))
            inputs = torch.from_numpy(inputs)
        elif phase == 2:
            inputs = self.blnk_vec
        else:
            inputs = self.blnk_vec

        self.current_index += batch
        mask = self.phase_oh[phase].unsqueeze(0)
        return inputs.float(), mask
    

    def getmask(self, batch=1):
        if self.current_index >= len(self.masks):
            self.make_new_graph()
        phase = self.masks[self.current_index]
        self.current_index += batch
        mask = self.phase_oh[phase].unsqueeze(0)
        return mask.cuda() if self.cuda is True else mask
    

    def getitem_combined(self, batch=1):
        inputs, mask = self.getitem(batch)
        # print("inputs: ", inputs)
        # print("mask: ", mask)
        combined = torch.cat([mask, inputs], 1)
        return combined.cuda() if self.cuda is True else combined
    

    def human_readable(self, ix, mask=None):
        print(PHASES[self.masks[self.current_index-1]], " Phases, " + str(self.current_index-1) + " Index")
        _input_vec = self.ix_to_vec(ix[0][4:])
        print("Input ::: from: ", _input_vec[0], " to: ", _input_vec[1])


def test_fn(data):
    phase_masks = data.make_new_graph()
    print("length of phase mask: ", len(phase_masks))
    for phase_idx in phase_masks:
        if phase_idx == 0 or phase_idx == 1 or phase_idx == 2:
            _t = data.getitem_combined()
            data.human_readable(_t.numpy())
        else:
            _t = data.getitem_combined()
            data.human_readable(_t.numpy().tolist())
            best_actions, all_actions = data.get_actions()
            data.update_state(best_actions[0])
            print("best action: ", best_actions)
            print("all action: ", all_actions)
            print("Current State: ", data.STATE)


# if __name__ == '__main__':
#     data = ShortestPathGraphData()
#     test_fn(data)