import yaml
from envs.gridworld import GridWorld
from agents.rl_agent import DQNAgent

def evaluate(policy_path):
    config = yaml.safe_load(open('config/default.yaml'))
    env = GridWorld(size=config['env']['size'])
    agent = DQNAgent(env.size*2, 4, config['agent']['learning_rate'])
    agent.load_state_dict(torch.load(policy_path))