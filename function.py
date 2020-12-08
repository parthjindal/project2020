def train():
    #checkout https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
    '''
    # suppose you first back-propagate loss1, then loss2 (you can also do the reverse)
    loss1.backward(retain_graph=True)
    loss2.backward() # now the graph is freed, and next process of batch gradient descent is ready
    optimizer.step() # update the network parameters
    '''

    pass

def validate():
    pass

def cal_accuracy():
    pass