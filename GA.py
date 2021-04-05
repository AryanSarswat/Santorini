from linear_rl_agents import LinearRlAgentV2
from Game import Board
from random import uniform
from tqdm import tqdm

def run_santorini(agent1,agent2):
    '''
    should run a game of Santorini, allow choice of AI/human players
    '''
    board = Board(agent1, agent2)
    win = None
    #initial worker placement
    board = board.PlayerA.place_workers(board)
    board = board.PlayerB.place_workers(board)
    current_player = 'A'
    
    def get_current_board_player(current_player):
        if current_player == 'A':
            return board.PlayerA
        else:
            return board.PlayerB
    #game loop
    while win == None:
        board_player = get_current_board_player(current_player)
        win = board.start_turn_check_win(board_player)
        if win != None:
            break
        else:
            """
            print(f'Current Player is {current_player}')
            board.print_board()
            print("----------------------------------------------------------------\n")
            """
            board = board_player.action(board)
            #because the board has been replaced, need to retrieve player obj again
            board_player = get_current_board_player(current_player)
            win = board.end_turn_check_win(board_player)
            if win != None:
                #board.print_board()
                break
        
        if current_player == 'A':
            current_player = 'B'
        else:
            current_player = 'A'
        
    return win



gen_100 = [0.14523219819780642, 1.6508246981208878, 4.641928472770049, 1.9166268804564177, -3.1362176308492065, -5.320305977518895, -4.195440163754433, 7.5790694380355905]
rootstrap_depth3_self_play_100_games = [-1.70041383, -1.40308437,  3.81622973,  0.98649831,  0.18495751, -4.61974509, -1.57060762,  1.29561011]


def train(init_weights):
    a = LinearRlAgentV2("A",3,trained_weights=init_weights)
    b = LinearRlAgentV2("B",3,trained_weights=init_weights)
    win = 0
    generations = 100
    num_games = 50

    for g in tqdm(range(generations)):
        win = 0
        for i in range(num_games):
            if "A" == run_santorini(agent1=a,agent2=b):
                win+=1
        if win > 0.5*num_games:
            temp_w = [k + uniform(-1,1) for k in a.trained_weights]
            b = LinearRlAgentV2("B",3,trained_weights=temp_w)
        else:
            temp_w = [k + uniform(-1,1) for k in b.trained_weights]
            a = LinearRlAgentV2("A",3,trained_weights=temp_w)

    if win > 0.5*num_games:
        print(f"At the end of {generations} generations the best weights is {a.trained_weights}")
        return a.trained_weights

    else:
        print(f"At the end of {generations} the best weights is {b.trained_weights}")
        return b.trained_weights

def test(test_weight):
    win = 0
    for i in range(100):
        if run_santorini(LinearRlAgentV2("A",3,trained_weights=test_weight),LinearRlAgentV2("B",3,trained_weights=rootstrap_depth3_self_play_100_games)) == "A":
            win+=1
    print(f"Win rate with new weights {win}%")

if __name__ == "__main__":
    new = train(gen_ten)
    test(new)


