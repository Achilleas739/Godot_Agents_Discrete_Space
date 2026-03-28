import numpy as np
from ActorCriticICM import MultiAgentSAC, AgentSubnet, HierarchicalAgent
from buffer import ReplayBuffer
from Wrapers import StableBaselinesGodotEnv
import torch

GREEN = "\033[92m"
RESET = "\033[0m"


def train(
    env,
    agents: dict,
    episodes=1000,
    max_episode_steps=300,
    warmup=20,
    use_checkpoints=True,
):
    total_num_steps = 0
    if use_checkpoints:
        for agent in agents.values():
            agent.load()

    total_rewards = dict()

    print(f"{GREEN}STARTED TRAINING{RESET}")
    for i_episode in range(episodes):
        episode_steps = 0
        done = False
        obs = env.reset()
        while episode_steps < max_episode_steps and not done:
            actions = []
            for state, policy_type, id in zip(
                obs["obs"], obs["policy_name"], obs["id"]
            ):
                agent: MultiAgentSAC = agents[policy_type]
                actor: AgentSubnet = agent.agent_policy
                memory: ReplayBuffer = actor.memory

                update = None

                if warmup > total_num_steps:
                    sample = (
                        env.action_spaces[id].sample()
                        if hasattr(env, "action_spaces")
                        else env.action_space.sample()
                    )
                    action_dict = sample
                    update = agent.warmup_update_parameters()
                    #update = None

                else:
                    action, _ = agent.select_action(state)
                    action_dict = {"movement": int(action[0])}
                    update = agent.update_parameters()

                actions.append(action_dict)

                if update is not None:
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        alpha_loss,
                        alpha,
                        entropy,
                        expected_q,
                        target_entropy,
                    ) = update

                    critics_counter = actor.critic_counter

                    agent.writer.add_scalar(
                        "loss/alpha_loss", alpha_loss, critics_counter
                    )
                    """
                    agent.writer.add_scalar(
                        "loss/prediction", prediction_loss, critics_counter
                    )
                    """
                    agent.writer.add_scalar("parameters/alpha", alpha, critics_counter)
                    agent.writer.add_scalars(
                        "parameters/entropy",
                        {"entropy": entropy, "target entropy": target_entropy},
                        critics_counter,
                    )

                    agent.writer.add_scalar(
                        "parameters/expected q value", expected_q, critics_counter
                    )

                    agent.writer.add_scalars(
                        "loss/critics",
                        {"critic_1": critic_1_loss, "critic_2": critic_2_loss},
                        critics_counter,
                    )

                    agent.writer.add_scalar(
                        "loss/Actor_" + f"{id}", policy_loss, critics_counter
                    )
            # print(f"{GREEN}actions {actions} {RESET}")

            next_obs, rewards, dones, _ = env.step(actions)
            episode_steps += 1
            total_num_steps += 1

            if episode_steps == max_episode_steps:
                masks = np.ones_like(dones, dtype=np.float32)

            else:
                masks = np.logical_not(dones).astype(np.float32)

            ids = [str(i) for i in range(len(obs["id"]))]

            for idx, (state, next_state, policy_type, id) in enumerate(
                zip(obs["obs"], next_obs["obs"], obs["policy_name"], ids)
            ):
                agent = agents[policy_type]
                memory = agent.agent_policy.memory
                action = actions[idx]["movement"]
                reward = rewards[idx]
                memory.store_transition(state, action, reward, next_state, masks[idx])
                agent.rewards[id] = agent.rewards.get(id, 0) + reward

            obs = next_obs

        for agent in agents.values():
            policy_type_rewards = agent.get_rewards().copy()
            total_rewards[agent.policy_name] = policy_type_rewards
            agent.writer.add_scalars("reward/train", policy_type_rewards, i_episode)
            agent.reset_rewards()
            if i_episode % 10 == 0 and i_episode > 0:
                print("Saving Models...")
                agent.save()
                print("Models Saved")

        print(
            f"Episode : {i_episode}, total num steps : {total_num_steps}, episode steps : {episode_steps}\nTotal rewards : \n{total_rewards}"
        )


def hierarchical_train(
    env,
    agents: dict,
    episodes=1000,
    max_episode_steps=300,
    warmup=20,
    use_checkpoints=True,
):
    total_num_steps = 0
    if use_checkpoints:
        for agent in agents.values():
            agent.load()

    total_rewards = dict()

    print(f"{GREEN}STARTED TRAINING{RESET}")
    for i_episode in range(episodes):
        episode_steps = 0
        done = False
        obs = env.reset()
        while episode_steps < max_episode_steps and not done:
            actions = []
            for state, policy_type, id in zip(
                obs["obs"], obs["policy_name"], obs["id"]
            ):
                agent: MultiAgentSAC = agents[policy_type]

                update = None

                if warmup > total_num_steps:
                    sample = (
                        env.action_spaces[id].sample()
                        if hasattr(env, "action_spaces")
                        else env.action_space.sample()
                    )
                    action_dict = sample

                else:
                    action = agent.select_action(state)
                    action_dict = {action[0]: int(action[1])}

                actions.append(action_dict)

            # print(f"{GREEN}actions {actions} {RESET}")

            next_obs, rewards, dones, _ = env.step(actions)
            episode_steps += 1
            total_num_steps += 1

            if episode_steps == max_episode_steps:
                masks = np.ones_like(dones, dtype=np.float32)

            else:
                masks = np.logical_not(dones).astype(np.float32)

            ids = [str(i) for i in range(len(obs["id"]))]

            # TODO: FIX MEMORY SYSTEM FOR THE NETS

            for idx, (state, next_state, policy_type, id) in enumerate(
                zip(obs["obs"], next_obs["obs"], obs["policy_name"], ids)
            ):
                agent: MultiAgentSAC = agents[policy_type]
                policy: HierarchicalAgent = agent.agent_policy
                action: dict = actions[idx]
                reward = rewards[idx]
                policy.memory_update(state, action, reward, next_state, masks[idx])
                # agent.agent_policy.reward += reward
                agent.rewards[id] = agent.rewards.get(id, 0) + reward

            obs = next_obs

        for agent in agents.values():
            policy_type_rewards = agent.get_rewards().copy()
            total_rewards[agent.policy_name] = policy_type_rewards
            agent.writer.add_scalars("reward/train", policy_type_rewards, i_episode)
            agent.reset_rewards()
            if i_episode % 1 == 0 and i_episode > 0:
                print("Saving Models...")
                agent.save()
                print("Models Saved")

        print(
            f"Episode : {i_episode}, total num steps : {total_num_steps}, episode steps : {episode_steps}\nTotal rewards : \n{total_rewards}"
        )


if __name__ == "__main__":
    # Hyper-parameters
    replay_buffer_size = int(1e6)
    episodes = 1000
    warmup = 3000
    batch_size = 128
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.6
    target_update_interval = 10
    learning_rate = [3e-4, 3e-4]
    icm_lr = 3e-4
    hidden_size = [256, 256]
    exploration_scaling_factor = 1.5
    max_episode_steps = 1000
    action_repeat = 10
    env_name = "Godot_Chase_Phase1"
    Path = r"C:\Users\cyach\OneDrive\Desktop\ML\Godot-discrete-ActionSpace\movement_test.exe"
    use_checkpoints = False
    hierarchical = False

    env = StableBaselinesGodotEnv(
        env_path=None,
        port=11008,
        show_window=False,
        seed=0,
        action_repeat=action_repeat,
        n_parallel=1,
        speed_up=0,
        max_episode_steps=max_episode_steps * action_repeat,
    )

    action_spaces = env.action_space

    print("Action space:", action_spaces)
    print("Observation space:", env.observation_space)

    obs = env.reset()
    print("Obs:", obs)

    observations = env.reset()

    observation_sizes = env.observation_space["obs"].shape[0]

    policy_types = []
    for i in observations["policy_name"]:
        if i in policy_types:
            continue
        else:
            policy_types += [i]

    agents = dict()

    for idx, type in enumerate(policy_types):
        agent = MultiAgentSAC(
            num_inputs=observation_sizes,
            action_space=action_spaces,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            hidden_size=hidden_size[idx],
            sac_lr=learning_rate[idx],
            icm_lr=icm_lr,
            agent_lr=learning_rate[idx],
            target_update_interval=target_update_interval,
            exploration_scaling_factor=exploration_scaling_factor,
            policy_name=type,
            batch_size=batch_size,
            hierarchical=hierarchical,
        )

        agents[type] = agent

    print("Agents ", agents)

    if hierarchical:
        print(f"{GREEN} Hierarchical train {RESET}")
        hierarchical_train(
            env,
            agents,
            max_episode_steps=max_episode_steps,
            use_checkpoints=use_checkpoints,
            warmup=warmup,
        )
    else:
        train(
            env,
            agents,
            max_episode_steps=max_episode_steps,
            use_checkpoints=use_checkpoints,
            warmup=warmup,
        )
