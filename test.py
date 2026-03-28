import numpy as np
from ActorCriticICM import MultiAgentSAC
from Wrapers import StableBaselinesGodotEnv

GREEN = "\033[92m"
RESET = "\033[0m"


def test(
    env,
    agents: dict,
    episodes=10,
    batch_size=64,
    max_episode_steps=300,
):
    for agent in agents.values():
        agent.load(evaluate=True)

    print(f"{GREEN}STARTED TESTING{RESET}")
    for i_episode in range(episodes):
        episode_steps = 0
        done = False
        obs = env.reset()
        total_rewards = dict()

        while episode_steps < max_episode_steps and not done:
            actions = []

            for state, policy_type, id in zip(
                obs["obs"], obs["policy_name"], obs["id"]
            ):
                agent: MultiAgentSAC = agents[policy_type]

                action, _ = agent.select_action(state)
                action_dict = {"movement": int(action[0])}
                actions.append(action_dict)

            next_obs, rewards, done_arr, _ = env.step(np.array(actions))
            done = bool(done_arr.any())
            episode_steps += 1

            ids = [str(i) for i in range(len(obs["id"]))]

            for idx, (policy_type, id) in enumerate(
                zip(obs["policy_name"], ids)
            ):
                agent = agents[policy_type]

                reward = rewards[idx]
                agent.rewards[id] = agent.rewards.get(id, 0) + reward
            
            obs = next_obs
            
        for agent in agents.values():
            policy_type_rewards = agent.get_rewards()
            total_rewards[agent.policy_name] = policy_type_rewards
            agent.reset_rewards()

        print(f"Episode : {i_episode} completed\n{GREEN}Total rewards : \n{total_rewards}{RESET}")


if __name__ == "__main__":
    # Hyper-parameters
    replay_buffer_size = int(1e6)
    episodes = 1000
    warmup = 20
    batch_size = 256
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    target_update_interval = 10
    learning_rate = 3e-4
    icm_lr = 3e-4
    hidden_size = [128, 128]
    exploration_scaling_factor = 1.5
    max_episode_steps = 1000
    action_repeat = 4
    env_name = "Godot_Chase_Phase1"
    Path = r"C:\Users\cyach\OneDrive\Desktop\ML\Godot-discrete-ActionSpace\movement_test.exe"
    Test = True
    
    env = StableBaselinesGodotEnv(
        env_path=Path,
        port=11008,
        show_window=True,
        seed=0,
        action_repeat=action_repeat,
        n_parallel=1,
        speed_up=1,
        max_episode_steps=max_episode_steps * action_repeat,
    )

    action_spaces = env.action_space

    obs = env.reset()

    observations = env.reset()

    observation_sizes = env.observation_space["obs"].shape[0]

    policy_types = []
    for i in observations["policy_name"]:
        if i in policy_types:
            continue
        else:
            policy_types += [i]

    unique_elements, counts = np.unique(observations["policy_name"], return_counts=True)
    policy_counts = dict(zip(unique_elements, counts))

    print(policy_counts)

    agents = dict()

    for idx, type in enumerate(policy_types):

        agent = MultiAgentSAC(
            num_inputs=observation_sizes,
            action_space=action_spaces,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            hidden_size=hidden_size[idx],
            sac_lr=learning_rate,
            icm_lr=icm_lr,
            agent_lr=learning_rate,
            target_update_interval=target_update_interval,
            exploration_scaling_factor=exploration_scaling_factor,
            policy_name=type,
            test = False
        )
        agents[type] = agent
    
    test(env, agents)