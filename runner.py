from Game import *

def run_santorini(agent1, agent2):
    '''
    should run a game of Santorini, allow choice of AI/human players
    '''
    board = Board(agent1, agent2)
    win = None
    #initial worker placement
    board = board.PlayerA.place_workers(board)
    board = board.PlayerB.place_workers(board)
    current_player = 'player_a'
    
    def get_current_board_player(current_player):
        if current_player == 'player_a':
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
            #board.print_board()
            #print("----------------------------------------------------------------\n")
            board = board_player.action(board)
            #because the board has been replaced, need to retrieve player obj again
            board_player = get_current_board_player(current_player)
            win = board.end_turn_check_win(board_player)
            if win != None:
                #board.print_board()
                break
        
        #swap players
        if current_player == 'player_a':
            current_player = 'player_b'
        else:
            current_player = 'player_a'
        
    return win

#run_santorini()