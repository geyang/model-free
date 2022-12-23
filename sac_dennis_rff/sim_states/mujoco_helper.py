def get_sim_state(env):
    return env.unwrapped.sim.data.qpos.copy(), env.unwrapped.sim.data.qvel.copy()


def set_sim_state(env, sim_state):
    qpos, qvel = sim_state
    env.unwrapped.sim.data.qpos[:] = qpos
    env.unwrapped.sim.data.qvel[:] = qvel
    return env.unwrapped._get_obs()


if __name__ == '__main__':
    import gym

    env = gym.make('Reacher-v2')
    obs = env.reset()

    sim_state = get_sim_state(env)
    obs_after = set_sim_state(env, sim_state)
    assert (obs - obs_after).sum() == 0, "the observation should remain the same"
