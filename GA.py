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


weights = [0,2,4,0,-2,-4,-1,1]
depth = 3
num_games = 50

a = LinearRlAgentV2("A",depth,trained_weights=weights)
b = LinearRlAgentV2("B",depth,trained_weights=weights)
win = 0
generations = 10


for g in tqdm(range(generations)):
    win = 0
    for i in range(num_games):
        if "A" == run_santorini(agent1=a,agent2=b):
            win+=1
    if win > 5:
        temp_w = [k + uniform(-1,1) for k in a.trained_weights]
        b = LinearRlAgentV2("B",depth,trained_weights=temp_w)
    else:
        temp_w = [k + uniform(-1,1) for k in b.trained_weights]
        a = LinearRlAgentV2("A",depth,trained_weights=temp_w)

print(win)

if win > 5:
    print(f"At the end of {generations} generations the best weights is {a.trained_weights}")

else:
    print(f"At the end of {generations} the best weights is {b.trained_weights}")
