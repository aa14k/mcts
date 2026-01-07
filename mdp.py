from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

State = Tuple[int, int]   # (x, y) with y increasing "up"
Action = str              # one of: "U", "D", "L", "R"


class GridWorldMDP:
    """
    Small stochastic GridWorld MDP compatible with the interface used in your MCTS snippet.

    Default layout (width=4, height=3), coordinates (x,y):
      y=2:  (0,2) (1,2) (2,2) (3,2)=+1 terminal
      y=1:  (0,1) (1,1)=WALL (2,1) (3,1)=-1 terminal
      y=0:  (0,0)=START (1,0) (2,0) (3,0)

    Dynamics:
    - Actions: U, D, L, R
    - Slip: intended direction with prob p_intended, perpendicular with remaining prob split evenly
      (U/D slip to L/R, L/R slip to U/D).
    - If a move hits a wall or boundary, you stay in place.
    - Reward is given on transition:
        * step_reward for non-terminal next states
        * terminal reward when you enter a terminal state (+1 or -1 by default)
    - Terminal states are absorbing (and have no actions).
    """

    ACTIONS: Tuple[Action, ...] = ("U", "D", "L", "R")

    def __init__(
        self,
        width: int = 4,
        height: int = 3,
        start: State = (0, 0),
        terminal_rewards: Optional[Dict[State, float]] = None,
        walls: Optional[Iterable[State]] = None,
        step_reward: float = -0.08,
        p_intended: float = 1.0,
        discount_factor: float = 0.95,
        seed: Optional[int] = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        if not (0.0 <= p_intended <= 1.0):
            raise ValueError("p_intended must be in [0, 1]")
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")

        self.width = width
        self.height = height
        self.start = start
        self.terminal_rewards: Dict[State, float] = (
            dict(terminal_rewards)
            if terminal_rewards is not None
            else {(3, 2): +1.0, (3, 1): -1.0}
        )
        self.walls: Set[State] = set(walls) if walls is not None else {(1, 1)}

        self.step_reward = float(step_reward)
        self.p_intended = float(p_intended)
        self.gamma = float(discount_factor)

        self._rng = random.Random(seed)

        # Optional gym-like stateful interface
        self._state: State = self.start

        self._validate()

    # ---- Robustness helpers (your pasted MCTS sometimes passes node instead of state) ----

    @staticmethod
    def _unwrap_state(state_or_node: Any) -> State:
        if hasattr(state_or_node, "state"):  # Node-like
            return state_or_node.state
        return state_or_node

    def _validate(self) -> None:
        if not self._in_bounds(self.start):
            raise ValueError("start is out of bounds")
        if self.start in self.walls:
            raise ValueError("start cannot be a wall")
        if self.start in self.terminal_rewards:
            raise ValueError("start cannot be terminal")

        for w in self.walls:
            if not self._in_bounds(w):
                raise ValueError(f"wall out of bounds: {w}")
            if w in self.terminal_rewards:
                raise ValueError(f"cell cannot be both wall and terminal: {w}")

        for t in self.terminal_rewards:
            if not self._in_bounds(t):
                raise ValueError(f"terminal out of bounds: {t}")

    def set_seed(self, seed: Optional[int]) -> None:
        self._rng = random.Random(seed)

    # ---- MDP interface expected by your MCTS code ----

    def get_initial_state(self) -> State:
        return self.start

    def get_discount_factor(self) -> float:
        return self.gamma

    def is_terminal(self, state_or_node: Any) -> bool:
        s = self._unwrap_state(state_or_node)
        return s in self.terminal_rewards

    def get_actions(self, state_or_node: Any) -> Set[Action]:
        s = self._unwrap_state(state_or_node)
        if self.is_terminal(s):
            return set()
        return set(self.ACTIONS)

    def get_transitions(self, state_or_node: Any, action: Action) -> List[Tuple[State, float]]:
        """
        Returns: [(next_state, probability), ...]
        This matches the textbook code's use (it uses this mainly for probability display).
        """
        s = self._unwrap_state(state_or_node)
        if self.is_terminal(s):
            return [(s, 1.0)]

        outcome_actions = self._stochastic_outcome_actions(action)
        probs: Dict[State, float] = defaultdict(float)

        for a, p in outcome_actions:
            ns = self._apply_action_deterministic(s, a)
            probs[ns] += p

        # stable ordering helps debugging
        return sorted(probs.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    def get_transition_outcomes(
        self, state_or_node: Any, action: Action
    ) -> List[Tuple[State, float, float, bool]]:
        """
        Full model outcomes:
          [(next_state, prob, reward, done), ...]
        Useful for value iteration and exact checks.
        """
        s = self._unwrap_state(state_or_node)
        if self.is_terminal(s):
            return [(s, 1.0, 0.0, True)]

        outcomes: List[Tuple[State, float, float, bool]] = []
        for ns, p in self.get_transitions(s, action):
            r = self.terminal_rewards.get(ns, self.step_reward)
            done = ns in self.terminal_rewards
            outcomes.append((ns, p, r, done))
        return outcomes

    def execute(self, state_or_node: Any, action: Action) -> Tuple[State, float, bool]:
        """
        Simulator step:
          (next_state, reward, done)
        """
        s = self._unwrap_state(state_or_node)
        outcomes = self.get_transition_outcomes(s, action)

        u = self._rng.random()
        cum = 0.0
        for ns, p, r, done in outcomes:
            cum += p
            if u <= cum + 1e-12:
                return ns, r, done

        # numerical fallback
        ns, _, r, done = outcomes[-1]
        return ns, r, done

    def all_states(self) -> List[State]:
        states: List[State] = []
        for x in range(self.width):
            for y in range(self.height):
                s = (x, y)
                if s in self.walls:
                    continue
                states.append(s)
        return states

    # ---- Internal transition logic ----

    def _in_bounds(self, s: State) -> bool:
        x, y = s
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_valid_cell(self, s: State) -> bool:
        return self._in_bounds(s) and (s not in self.walls)

    def _apply_action_deterministic(self, s: State, action: Action) -> State:
        x, y = s

        if action == "U":
            ns = (x, y + 1)
        elif action == "D":
            ns = (x, y - 1)
        elif action == "L":
            ns = (x - 1, y)
        elif action == "R":
            ns = (x + 1, y)
        else:
            raise ValueError(f"Unknown action: {action}")

        if not self._is_valid_cell(ns):
            return s
        return ns

    def _stochastic_outcome_actions(self, intended_action: Action) -> List[Tuple[Action, float]]:
        p = self.p_intended
        p_side = (1.0 - p) / 2.0

        if intended_action in ("U", "D"):
            return [(intended_action, p), ("L", p_side), ("R", p_side)]
        if intended_action in ("L", "R"):
            return [(intended_action, p), ("U", p_side), ("D", p_side)]

        raise ValueError(f"Unknown action: {intended_action}")

    # ---- Optional gym-like wrapper ----

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.set_seed(seed)
        self._state = self.start
        return self._state

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        ns, r, done = self.execute(self._state, action)
        self._state = ns
        return ns, r, done, {}


def value_iteration(
    mdp: GridWorldMDP, theta: float = 1e-12, max_iters: int = 100_000
) -> Tuple[Dict[State, float], Dict[Tuple[State, Action], float]]:
    """
    Exact DP oracle for this small MDP.
    Returns:
      V[s]
      Q[(s,a)]
    """
    states = mdp.all_states()
    V: Dict[State, float] = {s: 0.0 for s in states}
    gamma = mdp.get_discount_factor()

    for _ in range(max_iters):
        delta = 0.0
        for s in states:
            if mdp.is_terminal(s):
                continue

            best_q = float("-inf")
            for a in mdp.get_actions(s):
                q = 0.0
                for ns, p, r, _done in mdp.get_transition_outcomes(s, a):
                    q += p * (r + gamma * V[ns])
                best_q = max(best_q, q)

            new_v = best_q
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v

        if delta < theta:
            break

    Q: Dict[Tuple[State, Action], float] = {}
    for s in states:
        for a in mdp.get_actions(s):
            q = 0.0
            for ns, p, r, _done in mdp.get_transition_outcomes(s, a):
                q += p * (r + gamma * V[ns])
            Q[(s, a)] = q

    return V, Q


if __name__ == "__main__":
    mdp = GridWorldMDP(seed=0)  # default params
    V, Q = value_iteration(mdp)

    s0 = mdp.get_initial_state()
    actions = sorted(mdp.get_actions(s0))
    best_a = max(actions, key=lambda a: Q[(s0, a)])

    print("Initial state:", s0)
    print("Available actions:", actions)
    print("Optimal action at start (oracle):", best_a)
    print("Q(s0, a):", {a: round(Q[(s0, a)], 6) for a in actions})
