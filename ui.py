from tkinter import *
from tkinter import messagebox
import numpy as np
from constants import *
from maze_state import Qmaze
import datetime
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from training_experience import Experience

class MainWindow:
    def __init__(self, window):
        self.window = window
        self.frame = Frame(self.window, bg='#F2F2F2')
        Label(self.frame, text='M A Z E R U N N E R', font="Times 16",
              bg='#F2F2F2', fg='#020202').pack(side=TOP)

        maze_frame = Frame(self.frame, bg='#f2f2f2')
        self.maze = Maze(maze_frame)
        buttons_frame = Frame(self.frame)
        self.modify_maze_button = Button(buttons_frame, command=self.modify_maze,
                                    text='Modify Maze', relief=FLAT, bg='#6c3e5a', fg='#f2f2f2', padx=10, pady=10)
        self.train_model_button = Button(buttons_frame, command=self.train_model,
                                    text='Train Model', relief=FLAT, bg='#6c3e5a', fg='#f2f2f2', padx=10, pady=10)
        self.play_game_button = Button(buttons_frame, command=self.play_game,
                                    text='Play', relief=FLAT, bg='#6c3e5a', fg='#f2f2f2', padx=10, pady=10)
        self.modify_maze_button.pack(side=LEFT, fill=X, expand=True)
        self.train_model_button.pack(side=LEFT, fill=X, expand=True)
        self.play_game_button.pack(side=LEFT, fill=X, expand=True)
        buttons_frame.pack(side=BOTTOM, fill=X, anchor=S)
        maze_frame.pack(padx=30, pady=30, expand=True, fill=BOTH)
        self.maze.get_frame().pack(anchor=CENTER,)

    def modify_maze(self):
        if self.maze.is_locked:
            show_popup(
                "We'll unlock the maze now. Click on cells to create a wall.")
            self.train_model_button['state'] = DISABLED
            self.play_game_button['state'] = DISABLED
            self.maze.reset()
            self.maze.is_locked = False
            self.modify_maze_button.config(text="Lock Maze")
        else:
            show_popup("Locking the maze. You can now train or play.")
            self.train_model_button['state'] = NORMAL
            self.play_game_button['state'] = NORMAL
            self.maze.is_locked = True
            self.modify_maze_button.config(text="Modify Maze")

    def train_model(self):
        if not self.maze.is_locked:
            show_popup("Lock the maze before training.")
            return
        self.modify_maze_button['state'] = DISABLED
        self.play_game_button['state'] = DISABLED
        self.maze.draw_player()

    def play_game(self):
        qmaze = self.maze.maze
        self.maze.play_game(qmaze)


    def get_frame(self):
        return self.frame


class Maze():
    def __init__(self, window):
        self.width = self.height = 350
        self.canvas = Canvas(window, width=self.width+1,
                             height=self.height+1, borderwidth=0, highlightthickness=0)
        self.n = 5
        self.rows = self.n
        self.columns = self.n
        self.cell_width = self.width//self.n
        self.cell_height = self.height//self.n
        self.is_locked = True
        self.cell = {}
        # self.canvas.create_rectangle(0,0,10,10, fill="blue", tags="cell", )
        for column in range(self.n):
            for row in range(self.n):
                x1 = column*self.cell_width
                y1 = row*self.cell_height
                x2 = x1+self.cell_width
                y2 = y1+self.cell_height
                self.cell[row, column] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="grey", tags="cell", width=1,)
                self.canvas.tag_bind(
                    self.cell[row, column], "<Button-1>", self.cell_click)
        self.canvas.itemconfig(self.cell[self.n-1, self.n-1], fill="green")
        self.canvas.itemconfig(self.cell[0, 0], fill="red")
        self.maze = np.ones((self.n, self.n))
        self.model = self.build_model()

    def draw_player(self):
        self.canvas.create_rectangle(
            0, 0, self.cell_width, self.cell_height, fill="yellow", tags="player", outline="yellow", width=1)

    def redraw_player(self, row, column):
        x1 = column*self.cell_width
        y1 = row*self.cell_height
        x2 = x1+self.cell_width
        y2 = y1+self.cell_height
        self.canvas.coords("player", x1, y1, x2, y2)

    def reset(self):
        self.maze = np.ones((self.n, self.n))
        for column in range(self.n):
            for row in range(self.n):
                self.canvas.itemconfig(self.cell[row, column], fill="grey")
        self.canvas.itemconfig(self.cell[self.n-1, self.n-1], fill="green")
        self.canvas.itemconfig(self.cell[0, 0], fill="red")

    def cell_click(self, event):
        if self.is_locked:
            return
        # find cell
        cell_col = event.x//self.cell_width
        cell_row = event.y//self.cell_height
        if (cell_col == 0 and cell_row == 0) or (cell_col == self.n-1 and cell_row == self.n-1):
            return
        if(self.maze[cell_row, cell_col] == 0):
            self.maze[cell_row, cell_col] = 1.
            self.canvas.itemconfig(self.cell[cell_row, cell_col], fill="grey")
        else:
            self.maze[cell_row, cell_col] = 0.
            self.canvas.itemconfig(self.cell[cell_row, cell_col], fill="black")

    def qtrain(self, maze, **opt):
        self.draw_player()
        global epsilon
        n_epoch = opt.get('n_epoch', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 50)

        start_time = datetime.datetime.now()
        qmaze = Qmaze(maze)
        experience = Experience(self.model, max_memory=max_memory)
        win_history = []
        n_free_cells = len(qmaze.free_cells)
        hsize = qmaze.maze.size//2   # history window size
        win_rate = 0.0
        imctr = 1
        for epoch in range(n_epoch):
            loss = 0.0
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            self.redraw_player(rat_cell[0], rat_cell[1])
            game_over = False
                   # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()
            n_episodes = 0
            while not game_over:
                #a
                valid_actions = qmaze.get_valid_actions()
                if not valid_actions: break
                prev_envstate = envstate
                # Get next action
                if np.random.rand() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = np.argmax(experience.predict(prev_envstate))
                       # Apply action, get reward and new envstate
                envstate, reward, game_status = qmaze.act(action)
                rat_row, rat_col, _ = qmaze.state
                self.redraw_player(rat_row, rat_col)
                if game_status == 'win':
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False
                       # Store episode (experience)
                episode = [prev_envstate, action, reward, envstate, game_over]
                experience.remember(episode)
                n_episodes += 1
                       # Train neuself.ral network model
                inputs, targets = experience.get_data(data_size=data_size)
                h = self.model.fit(
                  inputs,
                  targets,
                  epochs=8,
                  batch_size=16,
                  verbose=0,
                )
                loss = self.model.evaluate(inputs, targets, verbose=0)
            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:]) / hsize

            dt = datetime.datetime.now() - start_time
            t = self.format_time(dt.total_seconds())
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
            print(template.format(epoch, n_epoch-1, loss,
                  n_episodes, sum(win_history), win_rate, t))
            # we simply check if training has exhausted all free cells and if in all
            # cases the agent won
            if win_rate > 0.9: epsilon = 0.05
            if sum(win_history[-hsize:]) == hsize and self.completion_check(qmaze):
                print("Reached 100%% win rate at epoch: %d" % (epoch,))
                break
    
    def play_game(self, qmaze, rat_cell = (0,0)):
        qmaze.reset(rat_cell)
        self.redraw_player(rat_cell[0],rat_cell[1])
        envstate = qmaze.observe()
        while True:
            prev_envstate = envstate
            q = self.model.predict(prev_envstate)
            action = np.argmax(q[0])

            envstate, reward, status = qmaze.act(action)
            rat_row, rat_col, _ = qmaze.state
            self.redraw_player(rat_row, rat_col)
            if status == 'win':
                return True
            elif status =='lose':
                return False

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.maze.size, input_shape=(self.maze.size,)))
        model.add(PReLU())
        model.add(Dense(self.maze.size))
        model.add(PReLU())
        model.add(Dense(num_actions))
        model.compile(optimizer='adam', loss='mse')
        return model

    def completion_check(self, qmaze):
        for cell in qmaze.free_cells:
            if not qmaze.get_valid_actions(cell):
                return False
            if not self.play_game(qmaze, cell):
                return False
        return True      

    def format_time(self, seconds):
        if seconds < 400:
            s = float(seconds)
            return "%.1f seconds" % (s,)
        elif seconds < 4000:
            m = seconds / 60.0
            return "%.2f minutes" % (m,)
        else:
            h = seconds / 3600.0
            return "%.2f hours" % (h,)
            
    def get_frame(self):
        return self.canvas


def center(obj):
    obj.update_idletasks()
    width = obj.winfo_width()
    height = obj.winfo_height()
    x = (obj.winfo_screenwidth()//2) - (width//2)
    y = (obj.winfo_screenheight()//2) - (height//2)
    obj.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def show_popup(msg):
    messagebox.showinfo("Information", msg)


root = Tk()
root.title('MazeRunner')
root.minsize(900, 512)
center(root)
ff = MainWindow(root)
ff.get_frame().pack(expand=True, fill=BOTH)
root.mainloop()
