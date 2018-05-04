# -*- coding: utf-8 -*-

from __future__ import print_function
import pickle
import re
import subprocess as sp
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
#from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class AiEngine(object):
    def __init__(self, name, path, width):
        self.name = name
        self.player = None
        self.co = sp.Popen(['engine/%s' % path], stdin=sp.PIPE, stdout=sp.PIPE,
            stderr=sp.PIPE, universal_newlines=True)
        self.write_command('start %i\n' % width)
        self.read_command(r'^OK$')
    
    def write_command(self, cmd):
        self.co.stdin.write(cmd)
        self.co.stdin.flush()
    
    def read_command(self, wanted):
        ep = re.compile(r'^error')
        wt = re.compile(wanted)
        while True:
            line = self.co.stdout.readline()
            if wt.match(line):
                return line
            if ep.match(line.lower()):
                print(line)
                return

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            if(board.last_move == -1):
                lx, ly = -1, -1
                self.write_command('board\ndone\n')
            else:
                lx = board.last_move // board.width
                ly = board.last_move % board.height
                cmd = 'turn {},{}\n'.format(lx, ly)
                self.write_command(cmd)
            text = self.read_command(r'^\d+,\d+$')
            loc = [int(n, 10) for n in text.split(',')]
            print('Alpha move : {},{} {} move : {},{}'.format(lx, ly, self.name, loc[0], loc[1]))
            move = board.location_to_move(loc)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "{} {}".format(self.name, self.player)
    

def run():
    w = 12
    try:
        ai_player = AiEngine('pela', 'pbrain-pela.exe', w)

        best_policy = PolicyValueNet(w, w, 'result/pytorch_12_5/current_policy.model')
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=2000)

        game = Game(Board(width=w, height=w, n_in_row=5))
        game.start_play(ai_player, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
