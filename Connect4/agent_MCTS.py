def agent_MCTS(obs, config):
    import random
    import time
    import math
    from copy import deepcopy
    import numpy as np

    class Configuration:
        PLAYERS = {'none': 0, 'one': 1, 'two': 2}
        OUTCOMES = {'none': 0, 'one': 1, 'two': 2, 'draw': 3}
        INF = math.inf
        def __init__(self, config):
            ROWS = config.rows
            COLS = config.columns
            INAROW = config.inarow

    class MCTSMeta:
        EXPLORATION = math.sqrt(2)
    
    Conf = Configuration(config)

    class State:
        def __init__(self, obs):
            self.board = np.asarray(obs.board).reshape(Conf.ROWS, Conf.COLS)
            self.player = obs.mark
            self.height = [Conf.ROWS - 1] * Conf.COLS
            self.last_played = []

        def get_board(self):
            return deepcopy(self.board)

        def move(self, col):
            self.board[self.height[col]][col] = self.player
            self.last_played = [self.height[col], col]
            self.height[col] -= 1
            self.player = Conf.PLAYERS['two'] if self.player == Conf.PLAYERS['one'] else Conf.PLAYERS['one']

        def get_legal_moves(self):
            return [col for col in range(Conf.COLS) if self.board[0][col] == 0]

        def check_win(self):
            if len(self.last_played) > 0 and self.check_win_from(self.last_played[0], self.last_played[1]):
                return self.board[self.last_played[0]][self.last_played[1]]
            return 0

        def check_win_from(self, row, column):
            player = self.board[row][column]
            columns = Conf.COLS
            rows = Conf.ROWS
            inarow = Conf.INAROW - 1
                
            def count(offset_row, offset_column):
                for i in range(1, inarow + 1):
                    r = row + offset_row * i
                    c = column + offset_column * i
                    if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or self.board[r][c] != player
                    ):
                        return i - 1
                return inarow

            return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
            )

        def game_over(self):
            return self.check_win() or len(self.get_legal_moves()) == 0

        def get_outcome(self):
            if len(self.get_legal_moves()) == 0 and self.check_win() == 0:
                return Conf.OUTCOMES['draw']

            return Conf.OUTCOMES['one'] if self.check_win() == Conf.PLAYERS['one'] else Conf.OUTCOMES['two']

    class Node:
        def __init__(self, move, parent):
            self.move = move
            self.parent = parent
            self.N = 0
            self.Q = 0
            self.children = {}
            self.outcome = Conf.PLAYERS['none']

        def add_children(self, children: dict) -> None:
            for child in children:
                self.children[child.move] = child

        def value(self, explore: float = MCTSMeta.EXPLORATION):
            if self.N == 0:
                return 0 if explore == 0 else Conf.INF
            else:
                return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)


    class MCTS:
        def __init__(self, state=State(), database = None):
            self.root_state = deepcopy(state)
            self.root = Node(None, None)
            self.run_time = 0
            self.node_count = 0
            self.num_rollouts = 0
            self.database = database

        def select_node(self) -> tuple:
            node = self.root
            state = deepcopy(self.root_state)

            while len(node.children) != 0:
                children = node.children.values()
                max_value = max(children, key=lambda n: n.value()).value()
                max_nodes = [n for n in children if n.value() == max_value]

                node = random.choice(max_nodes)
                state.move(node.move)

                if node.N == 0:
                    return node, state

            if self.expand(node, state):
                node = random.choice(list(node.children.values()))
                state.move(node.move)

            return node, state

        def expand(self, parent: Node, state: State) -> bool:
            if state.game_over():
                return False

            children = [Node(move, parent) for move in state.get_legal_moves()]
            parent.add_children(children)

            return True

        def roll_out(self, state: State) -> int:
            while not state.game_over():
                state.move(random.choice(state.get_legal_moves()))

            return state.get_outcome()

        def back_propagate(self, node: Node, turn: int, outcome: int) -> None:

            # For the current player, not the next player
            reward = 0 if outcome == turn else 1

            while node is not None:
                node.N += 1
                node.Q += reward
                node = node.parent
                if outcome == Conf.OUTCOMES['draw']:
                    reward = 0
                else:
                    reward = 1 - reward

        def search(self, time_limit: int):
            start_time = time.process_time()

            num_rollouts = 0
            while time.process_time() - start_time < time_limit:
                node, state = self.select_node()
                outcome = self.roll_out(state)
                if self.database:
                    def flatten_concatenation(matrix):
                        flat_list = []
                        for row in matrix:
                            flat_list += row
                        return " ".join(map(str, flat_list))
                    self.database.write(flatten_concatenation(state.board) + '\t' + str(outcome) + '\n')
                self.back_propagate(node, state.player, outcome)
                num_rollouts += 1

            run_time = time.process_time() - start_time
            self.run_time = run_time
            self.num_rollouts = num_rollouts

        def best_move(self):
            if self.root_state.game_over():
                return -1

            max_value = max(self.root.children.values(), key=lambda n: n.N).N
            max_nodes = [n for n in self.root.children.values() if n.N == max_value]
            best_child = random.choice(max_nodes)

            return best_child.move

        def move(self, move):
            if move in self.root.children:
                self.root_state.move(move)
                self.root = self.root.children[move]
                return

            self.root_state.move(move)
            self.root = Node(None, None)

        def statistics(self) -> tuple:
            return self.num_rollouts, self.run_time
        
    state = State(obs)
    mcts = MCTS(state)
    return mcts.best_move()