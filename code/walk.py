

class GridWalk:
    def __init__(self,column_length, row_length):
        self.nb_states = column_length * row_length
        self.row_length = row_length
        self.column_length = column_length
        self.num_actions = 4
        self.end_state = self.nb_states -1
        self.reset()

    def reset(self):
        self.current_state = 0
        return [0]

    def is_terminal(self):
        return self.current_state == self.end_state

    def get_position(self,state=None):
        if state is None :
            state = self.current_state
        return (state/self.row_length, state % self.row_length)

    def step(self,action):
        reward = 0
        # go up
        position = self.get_position()
        print('position',position)
        print('action',action)
        if action == 0 and position[0] > 0 :
            self.current_state -= self.row_length
        elif action == 1 and position[0] < self.column_length -1  : # go down
            self.current_state += self.row_length
        elif action == 2 and position[1] > 0  : #go left
            self.current_state -= 1
        elif action == 3 and position[1] < self.row_length -1  : #go right
            self.current_state += 1
        else :
            reward= -100
            print("invalid action")


        if self.is_terminal():
            reward = 1
            return [self.current_state], reward, True,{}
        return [self.current_state], reward, False,{}


class LineWalk:
    def __init__(self,nb_states):
        self.nb_states = nb_states
        self.num_actions = 2
        self.reset()

    def reset(self):
        self.current_state = 0
        return [0]

    def is_terminal(self):
        return self.current_state == self.nb_states -1

    def step(self,action):
        reward = -1
        if action == 0 :
            if self.current_state > 0:
                self.current_state -= 1
            else :
                reward = -2
        elif action == 1 :
            self.current_state += 1
            if self.is_terminal():
                reward = 100
                return [self.current_state], reward, True,None
        return [self.current_state], reward, False,None