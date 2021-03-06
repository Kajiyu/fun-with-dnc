import dnc_arity_list as dnc
import numpy as np
from utils import running_avg, flat, save, _variable
import utils as u
import torch
import torch.nn as nn
import torch.optim as optim
import time, random
from visualize import logger as sl
import argparse, os, time, json
import losses as L
from arg import args
from graph_problem_generator import ShortestPathGraphData

random.seed()
batch_size = 1
dnc_args = {
    'num_layers': 2,
    'num_read_heads': 4,
    'hidden_size': 250,
    'num_write_heads': 1,
    'memory_size':  100, #50
    'batch_size': batch_size
}


def generate_data_spec(args, solve=True):
    return {
        'num_nodes':args.n_max_nodes, 'batch_size':batch_size,
        'plan_phase':3, 'min_path_len':args.n_min_path,
        'max_path_len':args.n_max_path, 'cuda': args.cuda
    }


def setupDNC(args):
    """
        Loader for files or setup new DNC and optimizer
    :param args:
    :return:
    """
    data = ShortestPathGraphData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print('dnc_args:\n', dnc_args, '\n')
    if args.load == '':
        Dnc = dnc.DNC(**dnc_args)
        if args.opt == 'adam':
            optimizer = optim.Adam(Dnc.parameters(), lr=args.lr)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(Dnc.parameters(), lr=args.lr)
        else:
            optimizer = None
    else:
        model_path, optim_path = u.get_chkpt(args.load)
        print('loading', model_path)
        Dnc = dnc.DNC(**dnc_args)
        Dnc.load_state_dict(torch.load(model_path))

        optimizer = optim.Adam(Dnc.parameters(), lr=args.lr)
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path))

    if args.cuda is True:
        Dnc = Dnc.cuda()
    lstm_state = Dnc.init_rnn()
    return data, Dnc, optimizer, lstm_state


def tick(n_total, n_correct, truth, pred):
    n_total += 1
    n_correct += 1 if truth == pred else 0
    sl.global_step += 1
    return n_total, n_correct


def train_shortest_path_plan(args, data, DNC, lstm_state, optimizer):
    criterion = nn.CrossEntropyLoss().cuda() if args.cuda is True else nn.CrossEntropyLoss()
    cum_correct, cum_total, prob_times, n_success = [], [], [], 0
    penalty = 1.1

    for trial in range(args.iters):
        if trial % 100 == 0:
            ___score, ___goal_score = test_shortest_path_planning(args, data, DNC, lstm_state)
        start_prob = time.time()
        phase_masks = data.make_new_graph()
        print("Shortest Path :: ", data.shortest_path)
        n_total, n_correct, prev_action, loss, stats = 0, 0, None, 0, []
        dnc_state = DNC.init_state(grad=False)
        lstm_state = DNC.init_rnn(grad=False) # lstm_state, 
        optimizer.zero_grad()

        for phase_idx in phase_masks:

            if phase_idx == 0 or phase_idx == 1:
                inputs = _variable(data.getitem_combined())
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)
            
            elif phase_idx == 2:
                mask = _variable(data.getmask())
                inputs = torch.cat([mask, prev_action], 1)
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)
            
            else:
                best_nodes, all_nodes = data.get_actions()
                if not best_nodes:
                    break
                if args.zero_at == 'step':
                    optimizer.zero_grad()
                
                mask = data.getmask()
                prev_action = prev_action.cuda() if args.cuda is True else prev_action
                # print("previous action: ", prev_action)
                pr = u.depackage(prev_action)

                final_inputs = _variable(torch.cat([mask, pr], 1))
                logits, dnc_state, lstm_state = DNC(final_inputs, lstm_state, dnc_state)
                exp_logits = data.ix_input_to_ixs(logits)
                current_state = data.STATE
                guided = random.random() < args.beta
                sup_flag = None
                if guided: # guided loss
                    final_action, lstep = L.naive_loss_for_shortest_path(exp_logits, best_nodes, current_state, criterion, log=True)
                    sup_flag = "Yes"
                else: # pick own move
                    final_action, lstep = L.naive_loss_for_shortest_path(exp_logits, all_nodes, current_state, criterion, log=True)
                    sup_flag = "No"
                action_own = u.get_prediction(exp_logits)

                final_loss = lstep
                final_loss.backward(retain_graph=args.ret_graph)
                if args.clip:
                    torch.nn.utils.clip_grad_norm(DNC.parameters(), args.clip)
                optimizer.step()
                loss = lstep
                print("Supervised: "+sup_flag+", "+str(data.current_index)+" index, from: "+str(current_state)+", to: "+str(final_action)+", loss: ", final_loss.data[0])

                data.STATE = final_action

                prev_action = torch.from_numpy(np.array(data.vec_to_ix([current_state, final_action])).reshape((1, 61))).float()

        #### under experiment ####
        goal_loss = L.action_loss_for_shortest_path(exp_logits, data.goal, current_state, criterion, log=True)
        goal_loss.backward(retain_graph=args.ret_graph)
        optimizer.step()
        print("Goal Loss: ", goal_loss.data[0])
        ####
        end_prob = time.time()
        prob_times.append(start_prob - end_prob)
    # print("solved {} out of {} -> {}".format(n_success, args.iters, n_success / args.iters))
    return DNC, optimizer, lstm_state, 0.  # running_avg(cum_correct, cum_total)


def test_shortest_path_planning(args, data, DNC, lstm_state):
    criterion = nn.CrossEntropyLoss().cuda() if args.cuda is True else nn.CrossEntropyLoss()
    cum_correct, cum_total, prob_times, n_success = [], [], [], 0
    action_score = 0
    goal_score = 0
    total_actions = 0
    total_problems = 2
    print("\n")
    print("Accuracy Test is started ....")
    output_dict = {}
    for i in range(total_problems):
        start_prob = time.time()
        phase_masks = data.make_new_graph()
        print("Shortest Path :: ", data.shortest_path)
        n_total, n_correct, prev_action, loss, stats = 0, 0, None, 0, []
        dnc_state = DNC.init_state(grad=False)
        lstm_state = DNC.init_rnn(grad=False) # lstm_state, 

        input_history = []
        access_output_v=[]
        m=[]
        rw=[]
        ww=[]
        l=[]
        lw=[]
        uuu=[]

        for phase_idx in phase_masks:

            if phase_idx == 0 or phase_idx == 1:
                inputs = _variable(data.getitem_combined())
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)
                input_history.append(inputs.data[0].numpy().tolist())
                access_output_v.append(torch.squeeze(dnc_state[0].data).numpy().tolist())
                m.append(torch.squeeze(dnc_state[1].data).numpy().tolist())
                rw.append(torch.squeeze(dnc_state[2].data).numpy().tolist())
                ww.append(torch.squeeze(dnc_state[3].data).numpy().tolist())
                l.append(torch.squeeze(dnc_state[4].data).numpy().tolist())
                lw.append(torch.squeeze(dnc_state[5].data).numpy().tolist())
                uuu.append(torch.squeeze(dnc_state[6].data).numpy().tolist())

            
            elif phase_idx == 2:
                mask = _variable(data.getmask())
                inputs = torch.cat([mask, prev_action], 1)
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)
                input_history.append(inputs.data[0].numpy().tolist())
                access_output_v.append(torch.squeeze(dnc_state[0].data).numpy().tolist())
                m.append(torch.squeeze(dnc_state[1].data).numpy().tolist())
                rw.append(torch.squeeze(dnc_state[2].data).numpy().tolist())
                ww.append(torch.squeeze(dnc_state[3].data).numpy().tolist())
                l.append(torch.squeeze(dnc_state[4].data).numpy().tolist())
                lw.append(torch.squeeze(dnc_state[5].data).numpy().tolist())
                uuu.append(torch.squeeze(dnc_state[6].data).numpy().tolist())
            
            else:
                best_nodes, all_nodes = data.get_actions()
                if not best_nodes:
                    break
                mask = data.getmask()
                prev_action = prev_action.cuda() if args.cuda is True else prev_action
                pr = u.depackage(prev_action)

                final_inputs = _variable(torch.cat([mask, pr], 1))
                logits, dnc_state, lstm_state = DNC(final_inputs, lstm_state, dnc_state)
                input_history.append(final_inputs.data[0].numpy().tolist())
                access_output_v.append(torch.squeeze(dnc_state[0].data).numpy().tolist())
                m.append(torch.squeeze(dnc_state[1].data).numpy().tolist())
                rw.append(torch.squeeze(dnc_state[2].data).numpy().tolist())
                ww.append(torch.squeeze(dnc_state[3].data).numpy().tolist())
                l.append(torch.squeeze(dnc_state[4].data).numpy().tolist())
                lw.append(torch.squeeze(dnc_state[5].data).numpy().tolist())
                uuu.append(torch.squeeze(dnc_state[6].data).numpy().tolist())
                exp_logits = data.ix_input_to_ixs(logits)
                current_state = data.STATE
                guided = random.random() < args.beta
                final_action, lstep = L.naive_loss_for_shortest_path(exp_logits, all_nodes, current_state, criterion, log=True)
                print(str(data.current_index)+" index, from: "+str(current_state)+", to: "+str(final_action))
                if final_action in best_nodes:
                    action_score = action_score + 1
                total_actions = total_actions + 1
                action_own = u.get_prediction(exp_logits)
                data.STATE = final_action
                prev_action = torch.from_numpy(np.array(data.vec_to_ix([current_state, final_action])).reshape((1, 61))).float()
        if data.goal == final_action:
            goal_score = goal_score + 1
        end_prob = time.time()
        prob_times.append(start_prob - end_prob)
        p_output_dict = {}
        p_output_dict["inputs"] = input_history
        p_output_dict["access_outputs"] = access_output_v
        p_output_dict["m"] = m
        p_output_dict["rw"] = rw
        p_output_dict["ww"] = ww
        p_output_dict["l"] = l
        p_output_dict["lw"] = lw
        p_output_dict["u"] = uuu
        p_output_dict["phases"] = torch.squeeze(phase_masks).numpy().tolist()
        output_dict[i] = p_output_dict
    action_score = float(action_score) / float(total_actions)
    goal_score = float(goal_score) / float(total_problems)
    print("Accuracy Test is ended .... " + "action score: " + str(action_score) + ", goal score: " + str(goal_score))
    print("\n")
    with open("output.json", "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    return action_score, goal_score


def train_manager(args, train_fn):
    datspec = generate_data_spec(args)
    print('\nInitial Spec', datspec)

    _, DNC, optimizer, lstm_state = setupDNC(args)
    start_ents, score, global_epoch = args.n_init_start, 0, args.start_epoch
    print('\nDnc structure', DNC)

    for problem_size in range(10):
        data_spec = generate_data_spec(args)
        data_spec["num_nodes"] = args.n_max_nodes + problem_size
        data = ShortestPathGraphData(**data_spec)
        # score, goal_score = test_shortest_path_planning(args, data, DNC, lstm_state)
        # print("Action Score", score)
        # print("Goal Score", goal_score)
        for train_epoch in range(args.n_phases):
            ep_start = time.time()
            global_epoch += 1
            print("\nStarting Epoch {}".format(train_epoch))
            # if train_epoch == 0 and args.save != '':
            #     save(DNC, optimizer, lstm_state, args, 0)
            DNC, optimizer, lstm_state, score = train_fn(args, data, DNC, lstm_state, optimizer)
            if (train_epoch + 1) % args.checkpoint_every and args.save != '':
                save(DNC, optimizer, lstm_state, args, global_epoch)
            score, goal_score = test_shortest_path_planning(args, data, DNC, lstm_state)
            ep_end = time.time()
            ttl_s = ep_end - ep_start
            print('finished epoch: {}, ttl-time: {:0.4f}'.format(
                train_epoch, ttl_s
            ))
            print('finished epoch: {}, action score: {}, goal score: {}, ttl-time: {:0.4f}, time/prob: {:0.4f}'.format(
                train_epoch, score, goal_score, ttl_s, ttl_s / args.iters
            ))
            if score > args.passing:
                print('model_successful: {}, {} '.format(score, train_epoch))
                print('----------------------WOO!!--------------------------')
                passing = True
                break
        if score < args.passing:
            print("Problem "+str(problem_size)+" Learning was failed ......")
            return


def test_manager(args, test_fn):
    datspec = generate_data_spec(args)
    print('\nInitial Spec', datspec)
    _, DNC, optimizer, lstm_state = setupDNC(args)
    start_ents, score, global_epoch = args.n_init_start, 0, args.start_epoch
    print('\nDnc structure', DNC)
    data_spec = generate_data_spec(args)
    data_spec["num_nodes"] = args.n_max_nodes
    data = ShortestPathGraphData(**data_spec)
    score, goal_score = test_shortest_path_planning(args, data, DNC, lstm_state)
    print("Action Score", score)
    print("Goal Score", goal_score)


if __name__== "__main__":
    trainable_flag = False
    if trainable_flag:
        train_manager(args, train_shortest_path_plan)
    else:
        test_manager(args, test_shortest_path_planning)