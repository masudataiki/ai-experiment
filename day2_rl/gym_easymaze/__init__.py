from gymnasium.envs.registration import register

register(
    id='EasyMaze-v0',
    entry_point='gym_easymaze.envs:EasyMazeEnv',
)

# 演習3.4.5 で編集した迷路
register(
    id='EasyMazeModified-v0',
    entry_point='gym_easymaze.envs:EasyMazeEnv',
    kwargs={'variant': 'modified'},
)
