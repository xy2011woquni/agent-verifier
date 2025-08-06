import yaml
from envs.gridworld import GridWorld
from agents.rl_agent import RLTrainer
from verifier.formal_verifier import FormalVerifier

def main():
    config = yaml.safe_load(open('config/default.yaml'))
    env = GridWorld(size=config['env']['size'])
    trainer = RLTrainer(env, config)
    verifier = FormalVerifier(policy_fn=trainer.agent.forward, timeout_secs=config['verifier']['timeout_secs'])
    for ep in range(config['training']['episodes']):
        trainer.train_episode()
        if ep % 50 == 0:
            ok = verifier.verify(state_space=None)
            print(f"Episode {ep}: verifier result = {ok}")
if __name__ == "__main__":
    main()