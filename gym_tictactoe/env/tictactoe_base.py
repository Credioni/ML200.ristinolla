import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
import pandas as pd
import time
from enum import Enum
import random
from random import randint

from IPython.display import clear_output


class Player(Enum):
    """
    Player enum class
    - Constaints numerical value to present each player on board
    """
    X = 0.5     # Player
    O = 1       # Opponent aka GYM

    def opponent(self):
        return Player.O if self == Player.X else Player.X

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render: bool=False, human_player: bool=False):
        """
        TIC TAC TOE
            Playing against randomly playing opponent.

            Player/AI plays with X
            Gym plays with O

            Tic Tac Toe -board in a shape of 1-dimensional (9,)-array
             rendered as a shape of (3,3)
        """
        self._render = render   # Used to render after every move.

        # Episode variables - Resetable
        self._state  = None     # [0 for i in range(9)]
        self._done   = None     # If game has ended
        self._winner = None     # Winner of the game

        self._current_step = None
        self._current_player = None
        self._current_step_reward = None

        # PLayers
        self._human_player = human_player

        self._PLAYER = Player.X
        self._OPPONENT_player = Player.O

        # Rewards
        self._reward_win = 100
        self._reward_tie = -5
        self._reward_lose = -20
        self._reward_illegal_move = -2
        self._reward_base_step = 0

        # Gym space variables
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(1, 9),
            dtype=np.float32
            )

    def reset(self) -> np.array:
        """
        Called before every episode
        - Resets and formats enviroment variables for the episode
        """
        self._state = [0.0 for i in range(9)]
        self._done  = False
        self._winner = None
        self._current_player = self._PLAYER
        self._current_step = 0
        self._current_step_reward = 0
        self._current_action = None

        # Randomly decides the starting player
        if bool(random.getrandbits(1)):
            self._current_player = self._OPPONENT_player
            self.__opponent_action()
            self.__update_boardstate()

        return self._get_observation()

    def seed(self, seed=None) -> list:
        # Doesnt affect the env
        self.np_random, self._seed = seeding.np_random(seed)
        return [seed]

    def step(self, action:int) -> (np.array, float, bool, dict):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self._state is not None, "Call reset before using step method."

        self._current_action = action
        self._current_step_reward = self._reward_base_step #0

        # If winner is decided or game has ended
        if self._winner != None or self._done:
            self.__update_boardstate(self._current_action)
            self._done = True
            return self._get_observation(), self._current_step_reward, self._done, {}

        # Action phase #
        self._current_step += 1

        ## Player
        player_legal_move = self.__player_action(self._current_action)
        self.__update_boardstate(legal_move=player_legal_move)

        ## Opponent - GYM
        if not self._done and player_legal_move:
            # If Player won or made illegal move no actions from gym
            self.__opponent_action()
            self.__update_boardstate()

        # End phase #
        # Opponent player winning doesnt change the done state
        #  before agent has seen the move that opponent has made.
        #  -> delayed done.

        # Output variables
        observation = self._get_observation()
        reward = self._current_step_reward
        done   = self._done
        info   = {}

        return observation, reward, done, info

    def render(self, mode="human"):
        print("Agent - X  | Opponent - 0")

        game_board = np.array(self._state)
        game_board = pd.DataFrame(np.reshape(game_board, (3,3)))
        game_board = game_board.replace(0, '.')
        game_board = game_board.replace(0.5, 'X')
        game_board = game_board.replace(1, 'O')

        game_board = game_board.values.tolist()
        print(" _______")
        for (x, y, z) in game_board:
            print("| {} {} {} |".format(x, y, z))
        print(" ¯¯¯¯¯¯¯")
        #print(game_board)
        print("- - - - - - - ")

        print("Current step:", self._current_step)
        print("Step Reward:", self._current_step_reward, "\n")

        if self._done:
            print(" GAME ENDED ")
            if self._winner == None and self._done:
                print(" -- TIE -- ")
            elif self._winner == self._PLAYER and self._done:
                print("- YOU WON -")
            else:
                print("- YOU LOST -")


    def _get_observation(self) -> np.array:
        tmp_state = np.array(self._state, copy=True)
        return tmp_state

    def __update_boardstate(self, legal_move=True):

        """
        Updates enviroments stats in different game states
        - Updates winner, current_step_reward and done states.
        """
        done = True
        if self.__is_winner(self._PLAYER):
            # Player/AI won
            winner = self._PLAYER
            reward = self._reward_win
        elif self.__is_winner(self._OPPONENT_player):
            # Opponent/GYM won
            winner = self._OPPONENT_player
            reward = self._reward_lose
        elif self.__is_gameboard_full():
            # Board is full and no winners
            winner = None
            reward = self._reward_tie
        else:
            # Game is going on, moves left and no winners yet
            if legal_move:
                reward = self._reward_base_step
            else:
                reward = self._reward_illegal_move
            winner = None
            done   = False

        self._winner = winner
        self._current_step_reward = reward
        self._done = done

        if self._render and not self._human_player:
            self.render()

    def __player_action(self, action:int) -> bool:
        # Check legal action
        if self.__is_legal_action(action):
            if self._current_player == self._PLAYER:
                self._state[action] = self._PLAYER.value
                self._current_player = self._current_player.opponent()
                return True
        else:
            return False

    def __opponent_action(self):
        """
        Opponent aka. GYM action process.
        """
        # legal actions in a list
        avaible_actions =\
            [i for i in range(len(self._state)) if self._state[i]==0]

        if self._human_player:
            while True:
                time.sleep(0.2)
                self.render()
                print("avaible_actions are from top left to bottom right")
                print(">", avaible_actions)
                try:
                    action = int(input(">"))
                except:
                    pass
                clear_output(wait=True)

                if self.__is_legal_action(action):
                    self._state[action] = self._OPPONENT_player.value
                    self._current_player = self._current_player.opponent()
                    self.render()
                    break
        else:
            # Checks if there is avaible actions and taking action
            if len(avaible_actions) > 0:
                action = np.random.choice(avaible_actions)
                if self.__is_legal_action(action):
                    self._state[action] = self._OPPONENT_player.value
                    self._current_player = self._current_player.opponent()

    def __is_legal_action(self, action:int) -> bool:
        """
        Check if action is legal action
        """
        return True if self._state[action] == 0.0 else False

    def __is_winner(self, player:Player) -> bool:
        """
        Checking if there is a winner on the board
        """
        winner_check_indices = [
            # Horizontal
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            # Vertival
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            # Cross
            [0, 4, 8], [2, 4, 6],
        ]
        # Checking gameboard for winner
        for windices in winner_check_indices:
            similars = 0
            mark = player.value #self._state[windices[0]]
            if mark == 0.0: continue

            for widx in windices:
                if mark == self._state[widx]:
                    similars += 1
                else:
                    break
            # Winner found
            if similars >= 3:
                return True
        # No winners
        return False

    def __is_gameboard_full(self) -> bool:
        """
        Check if gameboard is full
        """
        free_spaces = [i for i in range(len(self._state)) if self._state[i]==0]
        return True if len(free_spaces) == 0 else False







if __name__ == "__main__":
    gym_env = TicTacToeEnv()
    gym_env.reset()
