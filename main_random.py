import numpy as np

from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

from agents import RandomAgent, SimpleAgent

np.random.seed(1)

# Configurations
NUM_AGENTS = 1
NUM_TRIALS = 5
HEIGHT = 20
WIDTH = 20

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment

TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
LocalGridObs = LocalObsForRailEnv(view_height=10, view_width=2, center=2)
env = RailEnv(width=WIDTH,
              height=HEIGHT,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=2, min_dist=8, max_dist=99999, seed=1),
              schedule_generator=complex_schedule_generator(),
              number_of_agents=NUM_AGENTS,
              obs_builder_object=TreeObservation)
env.reset()

env_renderer = RenderTool(env)


# Import your own Agent or use RLlib to train agents on Flatlpython src/main_random.py --train --num-episodes=10000 --prediction-depth=150 --eps=0.9998 --checkpoint-interval=100 --buffer-size=10000and
# As an example we use a random agent here


# Initialize the agent with the parameters corresponding to the environment and observation_builder
agent = RandomAgent(218, 5)

# Empty dictionary for all agent action
action_dict = dict()
print("Starting Training...")

for trials in range(1, NUM_TRIALS + 1):

    # Reset environment and get initial observations for all agents
    obs, info = env.reset()
    print(obs, info)
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_data["speed"] = 1 / (idx + 1)
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    # Run episode
    for step in range(500):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(obs[a])
            action_dict.update({a: action})
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]
        obs = next_obs.copy()
        if done['__all__']:
            break
    print('Trial No. {}\t Score = {}'.format(trials, score))
