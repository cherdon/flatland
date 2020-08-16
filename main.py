import numpy as np
import torch
import os
from collections import deque
import datetime

# Flatland Imports
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, random_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, random_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus

# Model Imports
from observations import RailObsForRailEnv
from predictions import ShortestPathPredictorForRailEnv
from preprocessing import ObsPreprocessor
from agents import DQNAgent
from utils.plot import plot_metric
from parser import res_parse, debug_parse


MAX_STEPS = 200
MAX_RAILS = 100


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def rail_setup(args, rail_type="sparse"):
    # Speed mapping
    speed_ration_map = {1.: 1}  # Fast passenger train
    if args.multi_speed:
        speed_ration_map = {1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

    # Rail Type (Complex, Sparse, Random)
    if rail_type == "sparse":
        rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
                                               seed=args.seed,
                                               grid_mode=args.grid_mode,
                                               max_rails_between_cities=args.max_rails_between_cities,
                                               max_rails_in_city=args.max_rails_in_city,
                                               )
        schedule_generator = sparse_schedule_generator(speed_ration_map)
    else:
        rail_generator = random_rail_generator(seed=args.seed
                                               )
        schedule_generator = random_schedule_generator(speed_ration_map)

    # Custom Observations
    prediction_builder = ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth)
    obs_builder = RailObsForRailEnv(predictor=prediction_builder)

    param = AttrDict({
        'malfunction_rate': args.malfunction_rate,
        'min_duration': args.min_duration,
        'max_duration': args.max_duration
    })

    # Building the environment
    env = RailEnv(
        width=args.width,
        height=args.height,
        rail_generator=rail_generator,
        random_seed=0,
        schedule_generator=schedule_generator,
        number_of_agents=args.num_agents,
        obs_builder_object=obs_builder,
        malfunction_generator_and_process_data=malfunction_from_params(parameters=param)
    )

    # Show render of simulation
    if args.render:
        env_renderer = RenderTool(
            env,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=800,
            screen_width=800)

    if args.render:
        return env, env_renderer, obs_builder
    else:
        return env, obs_builder


def main(args):
    # Setting up the Rail Network
    if args.render:
        env, env_renderer, obs_builder = rail_setup(args)
    else:
        env, obs_builder = rail_setup(args)

    preprocessor = ObsPreprocessor(MAX_RAILS, args.reorder_rails)

    # Setting up the agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dqn = DQNAgent(args, bitmap_height=MAX_RAILS * 3, action_space=2, device=device)


    # Initialise Values
    epsilon = args.start_eps
    railenv_action_dict = {}
    network_action_dict = {}
    # Metrics
    done_agents = deque(maxlen=args.window_size)            # Fraction of done agents over total agents
    all_rewards = deque(maxlen=args.window_size)            # Cumulative rewards over all trials
    norm_reward_metrics = deque(maxlen=args.window_size)    # Normalized cumulative rewards over all trials
    # Track means over windows of window_size episodes
    done_agents_metrics = []
    mean_rewards_metrics = []
    mean_norm_reward_metrics = []
    epsilons = [epsilon]
    crash = [False] * args.num_agents
    update_values = [False] * args.num_agents
    buffer_obs = [[]] * args.num_agents

    # Main loop
    for trial_no in range(args.num_trials):

        # Resetting values
        cumulative_reward = 0
        altmaps = [None] * args.num_agents
        altpaths = [[]] * args.num_agents
        buffer_rew = [0] * args.num_agents
        buffer_done = [False] * args.num_agents
        curr_obs = [None] * args.num_agents

        # Resetting environment TODO figure out what is maps [4, 41, 151]
        maps, info = env.reset()

        if args.render:
            env_renderer.reset()

        # Loop through time step
        for step in range(MAX_STEPS - 1):
            # Save a copy of maps at the beginning
            buffer_maps = maps.copy()
            # rem first bit is 0 for agent not departed
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                crash[a] = False
                update_values[a] = False
                network_action = None
                action = None

                # AGENT : ARRIVED
                if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
                    maps[a, :, :] = 0
                    network_action = 0
                    action = RailEnvActions.DO_NOTHING

                # AGENT : NOT DEPARTED
                elif agent.status == RailAgentStatus.READY_TO_DEPART:
                    update_values[a] = True
                    obs = preprocessor.get_obs(a, maps[a], buffer_maps)
                    curr_obs[a] = obs.copy()

                    # Epsilon-greedy action selection
                    q_values = dqn.act(obs).cpu().data.numpy()
                    if np.random.random() > epsilon:
                        network_action = np.argmax(q_values)
                    else:
                        network_action = np.random.choice([0, 1])

                    if network_action == 0:
                        action = RailEnvActions.DO_NOTHING
                    else: # Go
                        crash[a] = obs_builder.check_crash(a, maps)

                        if crash[a]:
                            network_action = 0
                            action = RailEnvActions.STOP_MOVING
                        else:
                            maps = obs_builder.update_bitmaps(a, maps)
                            action = obs_builder.get_agent_action(a)

                # Agent is entering a switch
                elif obs_builder.is_before_switch(a) and info['action_required'][a]:
                    # If the altpaths cache is empty or already contains
                    # the altpaths from the current agent's position
                    if len(altpaths[a]) == 0 or agent.position != altpaths[a][0][0].position:
                        altmaps[a], altpaths[a] = obs_builder.get_altmaps(a)

                    if len(altmaps[a]) > 0:
                        update_values[a] = True
                        altobs = [None] * len(altmaps[a])
                        q_values = np.array([])
                        for i in range(len(altmaps[a])):
                            altobs[i] = preprocessor.get_obs(a, altmaps[a][i], buffer_maps)
                            q_values = np.concatenate([q_values, dqn.act(altobs[i]).cpu().data.numpy()])

                        # Epsilon-greedy action selection
                        if np.random.random() > epsilon:
                            argmax = np.argmax(q_values)
                            network_action = argmax % 2
                            best_i = argmax // 2
                        else:
                            network_action = np.random.choice([0, 1])
                            best_i = np.random.choice(np.arange(len(altmaps[a])))

                        # Use new bitmaps and paths
                        maps[a, :, :] = altmaps[a][best_i]
                        obs_builder.set_agent_path(a, altpaths[a][best_i])
                        curr_obs[a] = altobs[best_i].copy()

                    else:
                        print('[ERROR] No possible altpaths episode: {} timestep: {} agent: {}'.format(trial_no, step, a))
                        network_action = 0

                    if network_action == 0:
                        action = RailEnvActions.STOP_MOVING
                    else:
                        crash[a] = obs_builder.check_crash(a, maps, is_before_switch=True)

                        if crash[a]:
                            network_action = 0
                            action = RailEnvActions.STOP_MOVING
                        else:
                            action = obs_builder.get_agent_action(a)
                            maps = obs_builder.update_bitmaps(a, maps, is_before_switch=True)

                # Agent is following a rail
                elif info['action_required'][a]:
                    crash[a] = obs_builder.check_crash(a, maps)

                    if crash[a]:
                        network_action = 0
                        action = RailEnvActions.STOP_MOVING
                    else:
                        network_action = 1
                        action = obs_builder.get_agent_action(a)
                        maps = obs_builder.update_bitmaps(a, maps)

                # No action_required
                else:
                    network_action = 1
                    action = RailEnvActions.DO_NOTHING
                    maps = obs_builder.update_bitmaps(a, maps)

                # Add next action for each agent in this time step
                network_action_dict.update({a: network_action})
                railenv_action_dict.update({a: action})

            # Completed Agents, Not Trials
            # Observation is computed from bitmaps while state is computed from env step (temporarily)
            _, reward, done, info = env.step(railenv_action_dict)

            if args.render:
                env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
                env_renderer.gl.save_image('tmp/frames/flatland_frame_{:04d}.png'.format(step))
                print("Saving render to: tmp/frames/flatland_frame_{:04d}.png".format(step))

            # If debugging
            if args.debug:
                for a in range(env.get_num_agents()):
                    debug = debug_parse(agent=a,
                                        info=info,
                                        env=env,
                                        network_action=network_action_dict,
                                        railenv_action=railenv_action_dict)
                    print(debug)

            # Training the agent
            if args.train:
                for a in range(env.get_num_agents()):
                    # If crash (and penalty is imposed) then store negative reward
                    if args.crash_penalty and crash[a]:
                        dqn.step(curr_obs[a], 1, args.crash_penalty, curr_obs[a], True)

                    if not args.switch2switch:
                        if update_values[a] and not buffer_done[a]:
                            next_obs = preprocessor.get_obs(a, maps[a], maps)
                            dqn.step(curr_obs[a], network_action_dict[a], reward[a], next_obs, done[a])

                    else:
                        if update_values[a] and not buffer_done[a]:
                            # If observation from a previous switch
                            if len(buffer_obs[a]) != 0:
                                dqn.step(buffer_obs[a], 1, buffer_rew[a], curr_obs[a], done[a])
                                buffer_obs[a] = []
                                buffer_rew[a] = 0

                            if network_action_dict[a] == 0:
                                dqn.step(curr_obs[a], 1, reward[a], curr_obs[a], False)
                            elif network_action_dict[a] == 1:
                                # Storing the observation and update at the next switch
                                buffer_obs[a] = curr_obs[a].copy()

                        # Cache reward only if we have an observation from a previous switch
                        if len(buffer_obs[a]) != 0:
                            buffer_rew[a] += reward[a]

                    # Now update the done cache to avoid adding experience many times
                    buffer_done[a] = done[a]

            for a in range(env.get_num_agents()):
                cumulative_reward += reward[a]

            if done['__all__']:
                break

        # End of trial
        epsilon = max(args.end_eps, args.eps_decay * epsilon)  # Decrease epsilon
        epsilons.append(epsilon)

        # Metrics
        # Recording done agents
        num_agents_done = 0
        for a in range(env.get_num_agents()):
            if done[a]:
                num_agents_done += 1

        # Average agents done
        done_agents.append(num_agents_done / env.get_num_agents())
        done_agents_metrics.append((np.mean(done_agents)))

        # Average reward
        all_rewards.append(cumulative_reward)
        mean_rewards_metrics.append(np.mean(all_rewards))

        # Average normalised reward
        normalized_reward = cumulative_reward / (env.compute_max_episode_steps(env.width, env.height) + env.get_num_agents())
        norm_reward_metrics.append(normalized_reward)
        mean_norm_reward_metrics.append(np.mean(norm_reward_metrics))

        # Print training results info
        checkpoint = False
        if trial_no != 0 and (trial_no + 1) % args.checkpoint_interval == 0:
            checkpoint = True
        response = res_parse(num_agents=env.get_num_agents(),
                             width=args.width,
                             height=args.height,
                             trial_no=trial_no,
                             done_agents=done_agents_metrics[-1],  # Fraction of done agents
                             mean_rewards=mean_rewards_metrics[-1],
                             mean_norm_rewards=mean_norm_reward_metrics[-1],
                             prev_done=(num_agents_done / args.num_agents)*100,
                             epsilon=epsilon,
                             checkpoint=checkpoint)
        print(response)

        # Saving the model (based on datetime)
        if args.train and trial_no != 0 and (trial_no + 1) % args.save_interval == 0:
            torch.save(dqn.qnetwork_local.state_dict(), os.path.join("pretrained",
                                                                     "{agent}agent-{trials}trials-{accuracy}-weights-{date}.pt".format(
                                                                         agent=args.num_agents,
                                                                         trials=trial_no+1,
                                                                         accuracy=done_agents_metrics[-1],
                                                                         date=str(datetime.datetime.now())
                                                                     )))
    torch.save(dqn.qnetwork_local.state_dict(), os.path.join("pretrained",
                                                             "{agent}agent-{trials}trials-{accuracy}-weights-{date}.pt".format(
                                                                     agent=args.num_agents,
                                                                     trials=args.num_trials,
                                                                     accuracy=done_agents_metrics[-1],
                                                                     date=str(datetime.datetime.now())
                                                                     )))

    return mean_rewards_metrics, mean_norm_reward_metrics, done_agents_metrics, epsilons


def render_frames(num_agents, width, height, pretrained_path):
    from parser import args
    args.plot = True
    args.train = False
    args.load_model = pretrained_path
    args.num_agents = num_agents
    args.width = width
    args.height = height
    args.num_trials = 1
    args.render = True

    # Printing Initial Parameters
    params = "Parameters"
    for k, v in vars(args).items():
        params += '\n{parameter}: {value}'.format(parameter=k, value=v)
    print(params)

    # Where to save models and plots
    if not os.path.exists("pretrained"):
        os.makedirs("pretrained")
    if not os.path.exists('plots'):
        os.makedirs('plots')

    mean_rewards_metrics, mean_norm_reward_metrics, done_agents_metrics, epsilons = main(args)

    # Plotting Metrics
    if args.plot:
        plot_metric(x=[i for i in range(args.num_trials)], y=mean_rewards_metrics,
                    title="Mean Rewards-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=mean_norm_reward_metrics,
                    title="Mean Normalised Rewards-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=done_agents_metrics,
                    title="Done Agents-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=epsilons,
                    title="Epsilon-{}".format(str(datetime.datetime.now())))


if __name__ == '__main__':
    from parser import args
    args.plot = True
    args.train = True
    args.num_agents = 2
    args.num_trials = 1000

    # Printing Initial Parameters
    params = "Parameters"
    for k, v in vars(args).items():
        params += '\n{parameter}: {value}'.format(parameter=k, value=v)
    print(params)

    # Where to save models and plots
    if not os.path.exists("pretrained"):
        os.makedirs("pretrained")
    if not os.path.exists('plots'):
        os.makedirs('plots')

    mean_rewards_metrics, mean_norm_reward_metrics, done_agents_metrics, epsilons = main(args)

    # Plotting Metrics
    if args.plot:
        plot_metric(x=[i for i in range(args.num_trials)], y=mean_rewards_metrics, title="Mean Rewards-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=mean_norm_reward_metrics, title="Mean Normalised Rewards-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=done_agents_metrics, title="Done Agents-{}".format(str(datetime.datetime.now())))
        plot_metric(x=[i for i in range(args.num_trials)], y=epsilons, title="Epsilon-{}".format(str(datetime.datetime.now())))
