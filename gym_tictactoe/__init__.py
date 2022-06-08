from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='gym_tictactoe.env:TicTacToeEnv',
    #kwargs={render=False, human_player=False},
)
