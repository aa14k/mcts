import torch
from collections import defaultdict
from tqdm import tqdm
from mdp import GridWorldMDP


mdp = GridWorldMDP(seed=0)
gamma = mdp.get_discount_factor()

state_to_node = {}
action_to_node = {}
node_to_action = {}
node_to_state = {}

rewards = {}

states = mdp.all_states()
node = 0
for state in states:
    state_to_node[state] = node
    node_to_state[node] = state
    node += 1

actions = mdp.get_actions(mdp.get_initial_state())  # all states have four actions
node = 0
for action in actions:
    action_to_node[action] = node
    node_to_action[node] = action
    node += 1

for state in states:
    for action in actions:
        rewards[state, action] = 0.0

max_state_nodes = len(states)
max_action_nodes = len(actions)

fully_expanded = torch.zeros(max_state_nodes, max_action_nodes)

Q = torch.zeros(max_state_nodes, max_action_nodes)
N = torch.zeros(max_state_nodes, max_action_nodes)
Prior = torch.nn.functional.softmax(torch.zeros(max_state_nodes, max_action_nodes), dim=-1)
policy = torch.distributions.Categorical(Prior)

# --- parents as requested: child -> list of (parent_state_node, parent_action_node)
state_parents = defaultdict(list)  # child_snode -> [(parent_snode, parent_anode), ...]

def record_parent(child_state, parent_state, parent_action):
    cs = state_to_node[child_state]
    ps = state_to_node[parent_state]
    pa = action_to_node[parent_action]
    state_parents[cs].append((ps, pa))

def argmax(tens: torch.Tensor) -> int:
    m = tens.max()
    idx = torch.where(tens == m)[0]
    j = torch.randint(0, idx.numel(), (1,), device=tens.device)
    return int(idx[j].item())

def is_fully_expanded(state):
    s = state_to_node[state]
    return torch.all(N[s] > 0).item()

def puct(state, cpuct=1.6):
    acts = mdp.get_actions(state)
    scores = torch.zeros(len(acts))
    s = state_to_node[state]
    for act in acts:
        a = action_to_node[act]
        if N[s, a] != 0:
            scores[a] = Q[s, a] + cpuct * Prior[s, a] * torch.sqrt(torch.sum(N[s])) / (1 + N[s, a])
        else:
            scores[a] = float("Inf")
    return argmax(scores)

def selection(state, max_steps=200):
    # returns: leaf_state, path = [(s0,a0), (s1,a1), ...]
    path = []
    steps = 0
    while (not mdp.is_terminal(state)) and is_fully_expanded(state):
        action = node_to_action[puct(state)]
        path.append((state, action))

        state_, rew, done = mdp.execute(state, action)
        record_parent(state_, state, action)
        rewards[state, action] = rew  # store immediate reward

        state = state_
        if done or steps > max_steps:
            break
        steps += 1
    return state, path

def expansion(state, path):
    # expand exactly one untried action if possible
    if mdp.is_terminal(state):
        return state, None, path

    s = state_to_node[state]
    untried = torch.where(N[s] == 0)[0]
    if untried.numel() == 0:
        return state, None, path

    # choose an untried action (random tie-break)
    j = torch.randint(0, untried.numel(), (1,)).item()
    a = int(untried[j].item())
    action = node_to_action[a]

    path.append((state, action))

    state_, rew, done = mdp.execute(state, action)
    record_parent(state_, state, action)
    rewards[state, action] = rew

    return state_, action, path

def simulation(state, max_steps=200):
    G = 0.0
    depth = 0
    for _ in range(max_steps):
        if mdp.is_terminal(state):
            break

        action = rollout_action_eps_greedy(state)                # CHANGED
        a = action_to_node[action]
        fully_expanded[state_to_node[state], a] += 1

        state_, rew, done = mdp.execute(state, action)
        record_parent(state_, state, action)
        rewards[state, action] = rew

        G += (gamma ** depth) * rew
        depth += 1
        state = state_
        if done:
            break
    return G

def backprop(path, G_leaf):
    # Standard MCTS backup along the visited edges.
    # If rollout return from leaf is G_leaf (already discounted from leaf),
    # then backing up one step needs: G = r(s,a) + gamma * G_child.
    G = G_leaf
    for state, action in reversed(path):
        r = rewards[state, action]
        G = r + gamma * G

        s, a = state_to_node[state], action_to_node[action]
        N[s, a] += 1
        Q[s, a] += (G - Q[s, a]) / N[s, a]

        if state == mdp.get_initial_state():
            break

def rollout_action_eps_greedy(state, eps=0.01):
    s = state_to_node[state]

    # explore
    if torch.rand(()) < eps:
        a = int(torch.randint(0, max_action_nodes, (1,)).item())
        return node_to_action[a]

    # exploit, but force trying unvisited actions first
    untried = torch.where(N[s] == 0)[0]
    if untried.numel() > 0:
        j = torch.randint(0, untried.numel(), (1,)).item()
        return node_to_action[int(untried[j].item())]

    # greedy w.r.t. current Q with random tie-break
    a = argmax(Q[s])
    return node_to_action[a]


steps = 10000
for t in tqdm(range(steps)):
    leaf, path = selection(mdp.get_initial_state())
    leaf2, action, path = expansion(leaf, path)
    G = simulation(leaf2,max_steps=200)
    backprop(path, G)

    if t % 200 == 0:
        s0 = state_to_node[mdp.get_initial_state()]
        print("Q(s0):", Q[s0], "N(s0):", N[s0])
