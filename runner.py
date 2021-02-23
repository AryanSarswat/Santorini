from Game import *


def main():
    win = None
    board = Board()
    board.intialize_workers([[1,1],[1,3]],[[3,1],[3,3]])
    Player = 1
    while win == None:
        win = board.start_turn_check_win(Player)
        if win != None:
            break
        else:
            board.print_board()
            print("----------------------------------------------------------------\n")
            #Establish location of Player_n workers
            if Player == 1:
                Player_n_workers = list(map(lambda Worker: Worker.current_location,board.Player_1_Workers))
            else:
                Player_n_workers = list(map(lambda Worker: Worker.current_location,board.Player_2_Workers))
            #Input worker to move coordinates and format into list
            worker_to_move_coord = input(f"Player {Player} Please the coordinates of the worker to move: ")
            worker_to_move_coord = list(map(lambda x: int(x),worker_to_move_coord.split(",")))
            #Error Handling for picking the worker
            while worker_to_move_coord not in Player_n_workers:
                print("That Worker does not belong to you \n")
                worker_to_move_coord = input(f"Player {Player} Please the coordinates of the worker to move: ")
                worker_to_move_coord = list(map(lambda x: int(x),worker_to_move_coord.split(",")))
            #Input the coordinates to move the selected worker
            new_coord = input(f"Player {Player} Please the coordinates where the worker is to be placed: ")
            new_coord = list(map(lambda x: int(x),new_coord.split(",")))
            #Error handling for new_coordinates for the selected worker
            while new_coord not in board.possible_worker_movements(worker_to_move_coord):
                print("That is not a valid move location")
                new_coord = input(f"Player {Player} Please the coordinates where the worker is to be placed: ")
                new_coord = list(map(lambda x: int(x),new_coord.split(",")))
            print("\n")
            #Update the worker's location and print new state of the board
            board.update_worker_location(worker_to_move_coord,new_coord)
            board.print_board()
            print("----------------------------------------------------------------\n")
            #Input the coordinate to build a building
            build_coord = input(f"Player {Player} Please enter coordinates of where you would like to build: ")
            build_coord = list(map(lambda x: int(x),build_coord.split(",")))
            #Error handling for the building coordinates
            while build_coord not in board.valid_building_options(new_coord):
                print("That is not valid build location \n")
                build_coord = input(f"Player {Player} Please enter coordinates of where you would like to build: ")
                build_coord = list(map(lambda x: int(x),build_coord.split(",")))  
            print("----------------------------------------------------------------\n")
            win = board.end_turn_check_win(Player)
            if win != None:
                board.print_board()
                break
            else:
                board.update_building_level(build_coord)
        if Player == 1:
            Player = 2
        else:
            Player = 1

    print(win)

if __name__=='__main__':
    main()