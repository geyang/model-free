# Takes NormalizedBoxEnv into account

def get_sim_state(env):
    return env.env.unwrapped.env.physics.data.qpos.copy(), env.env.unwrapped.env.physics.data.qvel.copy()


def set_sim_state(env, sim_state):
    qpos, qvel = sim_state
    raw_env = env.env.unwrapped.env
    raw_env.physics.data.qpos[:] = qpos
    raw_env.physics.data.qvel[:] = qvel
    return env.env._get_obs()


if __name__ == '__main__':
    import gym
    from sac_dennis_rff.utils import NormalizedBoxEnv

    env = NormalizedBoxEnv(gym.make('dmc:Walker-run-v1'))
    obs = env.reset()
    print(obs)

    sim_state = get_sim_state(env)
    obs_after = set_sim_state(env, sim_state)
    print(obs_after)
    assert (obs - obs_after).sum() == 0, "the observation should remain the same"
