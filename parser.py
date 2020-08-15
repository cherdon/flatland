import argparse


def res_parse(**kwargs):
    checkpoint = ""
    if kwargs['checkpoint']:
        checkpoint = "(Checkpoint)"
    parsed = '-------------------\n' \
             '\r{checkpoint}Trial: {trial_no}\n' \
             '-------------------\n' \
             '{num_agents} Agents running\n' \
             'Size of rail: ({width},{height})\n' \
             '\tMean done agents: {done_agents:.2f}\n' \
             '\tMean reward: {mean_rewards:.2f}\n' \
             '\tMean normalized reward: {mean_norm_rewards:.2f}\n' \
             'Epsilon: {epsilon:.2f}\n'.format(
                trial_no=kwargs['trial_no'],
                num_agents=kwargs["num_agents"],
                width=kwargs['width'],
                height=kwargs['height'],
                done_agents=kwargs['done_agents'],
                mean_rewards=kwargs['mean_rewards'],
                mean_norm_rewards=kwargs['mean_norm_rewards'],
                epsilon=kwargs['epsilon'],
                checkpoint=checkpoint
                )
    if kwargs['prev_done']:
        parsed += "Percentage done agents: {prev_done:.1f}%\n".format(prev_done=kwargs['prev_done'])
    return parsed


def debug_parse(**kwargs):
    parsed = "-------------------\n" \
             "Agent Info: {agent_no}\n" \
             "Status: {status}\n" \
             "Position: {position}\n" \
             "Target: {target}\n".format(
                agent_no=kwargs['agent'],
                status=kwargs['info']['status'][kwargs['agent']],
                position=kwargs['env'].agents[kwargs['agent']].position,
                target=kwargs['env'].agents[kwargs['agent']].target
                )
    if kwargs['env'].agents[kwargs['agent']].moving:
        parsed += "Moving speed: {moving_speed}\n".format(moving_speed=kwargs['info']['speed'][kwargs['agent']])
    else:
        parsed += "Not moving\n"
    parsed += "Action required? {action_bool}\n" \
              "Network action: {action_network}\n" \
              "Railenv action: {action_rail}\n".format(
                action_bool=kwargs['info']['action_required'][kwargs['agent']],
                action_network=kwargs['network_action'][kwargs['agent']],
                action_rail=kwargs['railenv_action'][kwargs['agent']]
                )
    return parsed


parser = argparse.ArgumentParser(description='Flatlands')

# Env parameters
parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
parser.add_argument('--width', type=int, default=20, help='Environment width')
parser.add_argument('--height', type=int, default=20, help='Environment height')
parser.add_argument('--num-agents', type=int, default=4, help='Number of agents in the environment')
parser.add_argument('--max-num-cities', type=int, default=2, help='Maximum number of cities where agents can start or end')
parser.add_argument('--seed', type=int, default=1, help='Seed used to generate grid environment randomly')
parser.add_argument('--grid-mode', type=bool, default=True, help='Type of city distribution, if False cities are randomly placed')
parser.add_argument('--max-rails-between-cities', type=int, default=2, help='Max number of tracks allowed between cities, these count as entry points to a city')
parser.add_argument('--max-rails-in-city', type=int, default=3, help='Max number of parallel tracks within a city allowed')
parser.add_argument('--malfunction-rate', type=int, default=2000, help='Rate of malfunction occurrence of single agent')
parser.add_argument('--min-duration', type=int, default=0, help='Min duration of malfunction')
parser.add_argument('--max-duration', type=int, default=0, help='Max duration of malfunction')
parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
parser.add_argument('--prediction-depth', type=int, default=150, help='Prediction depth for shortest path strategy, i.e. length of a path')
parser.add_argument('--multi-speed', action='store_true', help='Enable agents with fractionary speeds')

# Algo
parser.add_argument('--crash-penalty', action='store_true', default=-100, help='Negative reward when collision')
parser.add_argument('--reorder-rails', action='store_true', help='Change rails order in bitmaps')
parser.add_argument('--switch2switch', action='store_true', help='Train using only bitmaps where the agent is before a switch')
parser.add_argument('--checkpoint-interval', type=int, default=200, help='Interval of episodes for each print')

# Model Training
parser.add_argument('--model-id', type=str, default="ddqn-example", help="Model name/id")
parser.add_argument('--num-episodes', type=int, default=1000, help="Number of episodes to run")
parser.add_argument('--start-eps', type=float, default=1.0, help="Initial value of epsilon")
parser.add_argument('--end-eps', type=float, default=0.01, help="Lower limit of epsilon (i.e. can't decrease more)")
parser.add_argument('--eps-decay', type=float, default=0.998, help="Factor to decrease eps in eps-greedy")
parser.add_argument('--buffer-size', type=int, default=10000, help='Size of experience replay buffer (i.e. number of tuples')
parser.add_argument('--batch-size', type=int, default=512, help='Size of mini-batch for replay buffer')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--tau', type=float, default=1e-3, help='Interpolation parameter for soft update of target network weights')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for SGD')
parser.add_argument('--update-every', type=int, default=10, help='How often to update the target network')
# Model Saving
parser.add_argument('--train', action='store_true', help='Training with DQN Model or not (just running/running with pre-trained)')
parser.add_argument('--load-model', type=str, default=None, help="Loading Pre-trained model")
parser.add_argument('--save-interval', type=int, default=100, help='Interval in which the trained model is saved')

# Misc
parser.add_argument('--plot', action='store_true', help='Plot execution info')
parser.add_argument('--debug', action='store_true', help='Print debug info')
parser.add_argument('--render', action='store_true', help='Render map')
# parser.add_argument('--profile', action='store_true', help='Print a profiling of where the program spent most of its time')
# parser.add_argument('--print', action='store_true', help='Save internal representations as files')
parser.add_argument('--window-size', type=int, default=100, help='Number of episodes to consider for moving average when evaluating model learning curve')


args = parser.parse_args()
