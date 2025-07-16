import numpy as np
from ActorCriticICM import MultiAgentSAC, AgentPolicy
from buffer import ReplayBuffer
from Wrapers import StableBaselinesGodotEnv









def train(
    env,
    agents: dict,
    episodes=1000,
    max_episode_steps=300,
    warmup=20,
    use_checkpoints = False
):
    total_num_steps = 0
    updates = 0
    
    if use_checkpoints:
        for agent in agents.values():
            agent.load()
    
    print("STARTED TRAINING")
    for i_episode in range(episodes):
        
        episode_steps = 0
        done = False
        obs = env.reset()
        total_rewards = dict()
        while episode_steps < max_episode_steps and not done:
            actions = []
            for state, policy_type, id in zip(obs['obs'], obs['policy_name'], obs['id']):
                
                agent : MultiAgentSAC = agents[policy_type]
                actor: AgentPolicy = agent.policies[id]
                memory : ReplayBuffer = actor.memory
                
                
                if warmup > actor.actor_steps():
                    action = env.action_space.sample()

                else:
                    action = agent.select_action(id, state)

                actions.append(np.array(action))

                if memory.can_sample():
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        ent_loss,
                        prediction_loss,
                        alpha,
                    ) = agent.update_parameters( memory, id)
                    
                    
                    critics_counter = agent.critics_counter
                    
                    
                    agent.writer.add_scalar(
                        "loss/entropy", ent_loss, critics_counter
                    )
                    agent.writer.add_scalar(
                        "loss/prediction", prediction_loss, critics_counter
                    )
                    agent.writer.add_scalar(
                        "parameters/alpha", alpha, critics_counter
                    )
                    agent.writer.add_scalars(
                        "loss/critics",
                        {"critic_1": critic_1_loss, "critic_2": critic_2_loss},
                        critics_counter,
                    )
                    
                    agent.writer.add_scalar(
                    "loss/Actor_"+f'{id}', policy_loss, total_num_steps
                    )

            next_obs, rewards, done_arr, _ = env.step(np.array(actions))
            done = bool(done_arr.any())
            episode_steps += 1
            total_num_steps += 1
            updates += 1
            
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            for idx, (state, next_state, policy_type, id) in enumerate(zip(obs['obs'], next_obs['obs'], obs['policy_name'], obs['id'])):
                agent = agents[policy_type]
                memory = agent.policies[id].memory

                reward = rewards[idx]
                memory.store_transition(state, actions[idx], reward, next_state, mask)
                agent.policies[id].reward += reward

        for agent in agents.values():
            policy_type_rewards = agent.get_rewards()
            total_rewards[agent.policy_name] = policy_type_rewards
            agent.writer.add_scalars("reward/train", policy_type_rewards, i_episode)
            agent.reset_rewards()  
            if i_episode % 10 == 0 and i_episode > 0:
                print("Saving Models...")
                agent.save()
                print("Models saves")


        print(
            f"Episode : {i_episode}, total num steps : {total_num_steps}, episode steps : {episode_steps}\nTotal rewards : \n{total_rewards}"
        )







if __name__ == "__main__":
    # Hyper-parameters
    replay_buffer_size = int(1e6)
    episodes = 3000
    warmup = 100
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.05
    alpha = 0.1
    target_update_interval = 100
    learning_rate = 3e-4
    icm_lr = 3e-4
    hidden_size = [512, 512]
    exploration_scaling_factor = 1.5
    max_episode_steps = 300
    action_repeat = 4
    env_name = "Godot_Chase_Phase1"
    Path = r"C:\Users\cyach\OneDrive\Desktop\ML\Godot-RL-MultiAgent\Single_test_agent.exe"
    use_checkpoints = False

    env = StableBaselinesGodotEnv(
        env_path=Path,
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
    
    
    observation_sizes = env.observation_space['obs'].shape[0]
    
    policy_types = []
    for i in observations['policy_name']:
        if i in policy_types:
            continue
        else:
            policy_types+= [i]
    
    
    print(policy_types)
    
    unique_elements, counts = np.unique(observations["policy_name"], return_counts = True)
    policy_counts = dict(zip(unique_elements, counts))
    
    print(policy_counts)
    
    agents = dict()
    
    for idx,type in enumerate(policy_types):
        agent_ids = [j for j in range(policy_counts[type])]
        
        print("agent ids ",agent_ids)
        
        agent = MultiAgentSAC(
            agent_names = agent_ids,
            num_inputs = observation_sizes,
            action_space = action_spaces,
            gamma = gamma,
            tau = tau,
            alpha = alpha,
            hidden_size = hidden_size[idx],
            sac_lr = learning_rate,
            icm_lr = icm_lr,
            agent_lr = learning_rate,
            target_update_interval = target_update_interval,
            exploration_scaling_factor = exploration_scaling_factor,
            policy_name = type,
            )        
        
        agents[type] = agent
                
        for id in agent_ids:
            
            memory = ReplayBuffer(
                        replay_buffer_size,
                        input_shape = observation_sizes,
                        n_actions = env.action_space.shape[0],
                        batch_size = batch_size
                    )
            
            agent.policies[id].memory = memory
        

    
    print("Agents ",agents)
    
    train(env, agents, max_episode_steps = max_episode_steps)






