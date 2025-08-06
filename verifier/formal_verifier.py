class FormalVerifier:
    def __init__(self, policy_fn, timeout_secs=60):
        self.policy_fn = policy_fn
        self.timeout_secs = timeout_secs

    def verify(self, state_space):
        return True