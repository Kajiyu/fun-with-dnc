import argparse
import dnc_arity_list as dnc
import numpy as np
from utils import running_avg, flat, repackage, dnc_checksum, save
import utils as u
import torch
import torch.nn as nn
import generators_v2 as gen
import torch.optim as optim
import time, random
from torch.autograd import Variable
import logger as sl
import os
import losses as L
import training as tfn
random.seed()


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--act', nargs='?', type=str, default='train', help='[]')
parser.add_argument('--load', nargs='?', type=str, default='', help='load model and state')
parser.add_argument('--save', nargs='?', type=str, default='', help='save if true')
parser.add_argument('--beta', nargs='?', type=float, default=0.8, help='mixture param from paper')
parser.add_argument('--log', nargs='?', type=int, default=1, help='summaries in tb')
parser.add_argument('--typed', nargs='?', type=int, default=1, help='summaries in tb')
parser.add_argument('--max_ents', nargs='?', type=int, default=6, help='summaries in tb')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--algo', type=str, default='dnc')
parser.add_argument('--feed_last', type=int, default=1, help='')

parser.add_argument('--opt_at', type=str, default='problem', help='')
parser.add_argument('--zero_at', type=str, default='step', help='')

parser.add_argument('--ret_graph', type=int, default=1, help='')
parser.add_argument('--rpkg_step', type=int, default=1, help='')

    # parser.add_argument('--num_tests', type=int, default=0, help='')
parser.add_argument('--num_tests', type=int, default=2, help='')
parser.add_argument('--num_repeats', type=int, default=2, help='')
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--checkpoint_every', type=int, default=1000, help='')
parser.add_argument('--n_phases', type=int, default=15)
parser.add_argument('--n_cargo', type=int, default=2)
parser.add_argument('--n_plane', type=int, default=2)
parser.add_argument('--n_airport', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5) #1e-5 in paper.
parser.add_argument('--iters', nargs='?', type=int, default=21, help='number of iterations')
args = parser.parse_args()

#

batch_size = 1
start_timer = time.time()
start = '{:0.0f}'.format(start_timer)
PASSING = 0.85


dnc_args = {'num_layers': 2, 'num_read_heads': 2, 'hidden_size': 250,
            'num_write_heads': 1, 'memory_size': 100, 'batch_size': batch_size}


def generate_data_spec(args, num_ents=2, solve=True):
    if args.typed is 1:
        ix_size = [4, args.max_ents, 4, args.max_ents, 4, args.max_ents]
        encoding = 2
    else:
        ix_state = args.max_ents * 3
        ix_size = [ix_state, ix_state, ix_state]
        encoding = 1
    
    return {'num_plane': num_ents, 'num_cargo': num_ents, 'num_airport': num_ents,
            'one_hot_size': ix_size, 'plan_phase': num_ents * 3,
            'batch_size': 1, 'encoding': encoding, 'solve': solve, 'mapping': None}


def setupLSTM(args):
    data = gen.AirCargoData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print(dnc_args)
    lr = 5e-5 if args.lr is None else args.lr
    # input_size = self.output_size + word_len * num_read_heads
    Dnc = dnc.VanillaLSTM(batch_size=1, num_layers=2, input_size=data.nn_in_size,
                          output_size=data.nn_out_size, hidden_size=250, num_reads=2)
    previous_out, (ho1, hc1), (ho2, hc2) = Dnc.init_state()
    if args.opt == 'adam':
        optimizer = optim.Adam([{'params': Dnc.parameters()}, {'params': ho1},
                                {'params': hc1}, {'params': hc2}], lr=lr)
    else:
        optimizer = optim.SGD([{'params': Dnc.parameters()}, {'params': ho1},
                               {'params': hc1}, {'params': hc2}], lr=lr)

    dnc_state = (previous_out, (ho1, hc1), (ho2, hc2))
    return data, Dnc, optimizer, dnc_state


def setupDNC(args):
    if args.algo == 'lstm':
        return setupLSTM(args)
    data = gen.AirCargoData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print('dnc_args:\n', dnc_args, '\n')
    if args.load == '':
        Dnc = dnc.DNC(**dnc_args)
        optimizer = optim.Adam([{'params': Dnc.parameters()}], lr=args.lr)
    else:
        model_path = args.prefix + '/models/dnc_model_' + args.load
        print('loading', model_path)
        Dnc = dnc.DNC(**dnc_args)
        Dnc.load_state_dict(torch.load(model_path))

        optim_path = args.prefix + '/models/dnc_model_' + args.load
        optimizer = optim.Adam([{'params': Dnc.parameters()}], lr=args.lr)
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path))
            optimizer.state = defaultdict(dict, optimizer.state)

    o, m, r, w, l, lw, u, (ho, hc) = Dnc.init_state()
    dnc_state = (o, m, r, w, l, lw, u, (ho, hc))
    return data, Dnc, optimizer, dnc_state


def tick(n_total, n_correct, truth, pred):
    n_total += 1
    n_correct += 1 if truth == pred else 0
    sl.global_step += 1
    return n_total, n_correct


def train_qa2(args, data, DNC, optimizer):
    """
        I am jacks liver. This is a sanity test

        0 - describe state.
        1 - describe goal.
        2 - do actions.
        3 - ask some questions
        :param args:
        :return:
        """
    criterion = nn.CrossEntropyLoss()
    cum_correct, cum_total = [], []

    for trial in range(args.iters):
        phase_masks = data.make_new_problem()
        n_total, n_correct, loss = 0, 0, 0
        dnc_state = DNC.init_state(grad=False)
        optimizer.zero_grad()

        for phase_idx in phase_masks:
            if phase_idx == 0 or phase_idx == 1:
                inputs = Variable(data.getitem_combined())
                logits, dnc_state = DNC(inputs, dnc_state)
            else:
                final_moves = data.get_actions(mode='one')
                if final_moves == []:
                    break
                data.send_action(final_moves[0])
                mask = data.phase_oh[2].unsqueeze(0)
                inputs2 = Variable(torch.cat([mask, data.vec_to_ix(final_moves[0])], 1))
                logits, dnc_state = DNC(inputs2, dnc_state)

                for _ in range(args.num_tests):
                    # ask where is ---?
                    if args.zero_at == 'step':
                        optimizer.zero_grad()
                    masked_input, mask_chunk, ground_truth = data.masked_input()
                    logits, dnc_state = DNC(Variable(masked_input), dnc_state)
                    expanded_logits = data.ix_input_to_ixs(logits)

                    #losses
                    lstep = L.action_loss(expanded_logits, ground_truth, criterion, log=True)
                    if args.opt_at == 'problem':
                        loss += lstep
                    else:
                        lstep.backward(retain_graph=args.ret_graph)
                        optimizer.step()
                        loss = lstep

                    #update counters
                    prediction = u.get_prediction(expanded_logits, [3, 4])
                    n_total, n_correct = tick(n_total, n_correct, mask_chunk, prediction)

        if args.opt_at == 'problem':
            loss.backward(retain_graph=args.ret_graph)
            optimizer.step()
            sl.writer.add_scalar('losses.end', loss.data[0], sl.global_step)

        cum_total.append(n_total)
        cum_correct.append(n_correct)
        sl.writer.add_scalar('recall.pct_correct', n_correct / n_total, sl.global_step)
        print("trial: {}, step:{}, accy {:0.4f}, cum_score {:0.4f}, loss: {:0.4f}".format(
            trial, sl.global_step, n_correct / n_total, running_avg(cum_correct, cum_total), loss.data[0]))
    return DNC, optimizer, dnc_state, running_avg(cum_correct, cum_total)


def train_plan(args, data, DNC, optimizer):
    """
        Things to test after some iterations:
         - on planning phase and on

         with goals - chose a goal and work toward that
        :param args:
        :return:
        """
    criterion = nn.CrossEntropyLoss()
    cum_correct, cum_total = [], []
    n_success = 0

    for trial in range(args.iters):

        phase_masks = data.make_new_problem()
        n_total, n_correct, prev_action, loss, stats = 0, 0, None, 0, []
        dnc_state = DNC.init_state(grad=False)
        optimizer.zero_grad()

        for phase_idx in phase_masks:

            if phase_idx == 0 or phase_idx == 1:
                inputs = Variable(data.getitem_combined())
                logits, dnc_state = DNC(inputs, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)

            elif phase_idx == 2:
                mask = data.getmask()
                # print(prev_action)
                inputs = Variable(torch.cat([mask, prev_action.data], 1))
                logits, dnc_state = DNC(inputs, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)

            else:
                # sample from best moves
                targets_star = data.get_actions(mode='best')
                all_actions = data.get_actions(mode='all')
                if targets_star == []:
                    break
                if args.zero_at == 'step':
                    optimizer.zero_grad()

                mask = data.getmask()
                final_inputs = Variable(torch.cat([mask, u.depackage(prev_action)], 1))
                logits, dnc_state = DNC(final_inputs, dnc_state)
                exp_logits = data.ix_input_to_ixs(logits)

                guided = random.random() < args.beta
                if guided: # guided loss
                    final_action, lstep = L.naive_loss(exp_logits, targets_star, criterion, log=True)
                else: # pick own move
                    final_action, lstep = L.naive_loss(exp_logits, all_actions, criterion, log=True)

                if args.opt_at == 'problem':
                    loss += lstep
                else:
                    lstep.backward(retain_graph=args.ret_graph)
                    optimizer.step()
                    loss = lstep

                data.send_action(final_action)
                action_own = u.get_prediction(exp_logits)
                if (trial + 1) % 20 == 0:
                    action_accs = u.human_readable_res(data, all_actions, targets_star,
                                                       final_action, action_own, guided, lstep.data[0])
                    stats.append(action_accs)
                n_total, _ = tick(n_total, n_correct, action_own, flat(final_action))
                # print([flat(t) for t in targets_star] )
                #print('OWN  ', action_own)
                n_correct += 1 if action_own in [tuple(flat(t)) for t in targets_star] else 0

                # feed own outputs, or feed final action?
                prev_action = data.vec_to_ix(final_action)

        if stats:
            arr = np.array(stats)
            correct = len([1 for i in list(arr.sum(axis=1)) if i == len(stats[0])]) / len(stats)
            sl.log_acc(list(arr.mean(axis=0)), correct)
            # print(list(arr.mean(axis=0)), )

        if args.opt_at == 'problem':
            floss = loss / n_total
            floss.backward(retain_graph=args.ret_graph)
            optimizer.step()
            sl.writer.add_scalar('losses.end', floss.data[0], sl.global_step)

        n_success += 1 if n_correct / n_total > PASSING else 0
        cum_total.append(n_total)
        cum_correct.append(n_correct)
        sl.writer.add_scalar('recall.pct_correct', n_correct / n_total, sl.global_step)
        print("trial {}, step {} trial accy: {}/{}, {:0.4f}, running total {}/{}, running avg {}, loss {:0.4f}  ".format(
            trial, sl.global_step, n_correct, n_total, n_correct / n_total, n_success, trial,
            running_avg(cum_correct, cum_total), loss.data[0]
            ))

    print("solved {} out of {} -> {}".format(n_success, args.iters, n_success / args.iters))
    return DNC, optimizer, dnc_state, running_avg(cum_correct, cum_total)


def train_manager(args, train_fn):
    datspec = generate_data_spec(args)
    print('\n', datspec)

    _, DNC, optimizer, dnc_state = setupDNC(args)
    start_ents = 2
    print(DNC)

    for problem_size in range(args.max_ents):
        test_size = problem_size + start_ents
        passing = False
        data_spec = generate_data_spec(args, num_ents=test_size, solve=test_size * 3)
        data_loader = gen.AirCargoData(**data_spec)

        print("beginning new training Size: {}".format(test_size))
        for train_epoch in range(args.n_phases):
            print("epoch {}, chksum {}".format(train_epoch, dnc_checksum(dnc_state)))

            DNC, optimizer, dnc_state, score = train_fn(args, data_loader, DNC, optimizer)
            id_str = '_s{}_ep_{}_gl_{}'.format(test_size, train_epoch, sl.global_step)
            save(DNC, optimizer, start, args, id_str)

            print('finished training epoch {}, score:{}, file {}{}'.format(train_epoch, score))
            if score > PASSING:
                print('model_successful: {}, {} '.format(score, train_epoch))
                passing = True
                break

        if passing == False:
            print("Training has failed for problem of size {}, after {} epochs of {} phases".format(
                test_size, args.max_ents, args.n_phases
            ))
            print("final score was {}".format(score))
            break


# python run.py --act qa --iters 10000 --opt adam --save _adam_typed_qa_long_
# python run.py --act dag --max_ents 6 --opt adam --save _new_hidden --ret_graph False --opt_at step
if __name__=="__main__":

    args.repakge_each_step = True if args.rpkg_step == 1 else False
    args.ret_graph = True if args.ret_graph == 1 else False
    sl.log_step += args.log

    args.prefix = '/output/' if args.env == 'floyd' else './'
    if not os.path.exists(args.prefix + 'models'):
        os.mkdir(args.prefix + 'models')

    if args.save != '':
        f = open(args.prefix + 'models/params_' + str(start) + str(args.save) + '.txt', 'w')
        f.write(str(args))
        f.close()

    print(args)
    if args.act == 'plan':
        train_manager(args, train_plan)
    elif args.act == 'qa':
        train_manager(args, train_qa2)
    elif args.act == 'clean':
        pass
    else:
        print("wrong action")