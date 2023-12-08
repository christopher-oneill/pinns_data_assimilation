

import numpy as np
import torch
import matplotlib.pyplot as plot



x = np.float64(6*np.random.rand(100,1)-3)

x = torch.from_numpy(x)

# test function
y = -np.power(x,3)+2*np.power(x,3)+x-2

class ResidualLayer(torch.nn.Module):
    def __init__(self,size_in,size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        weights = torch.zeros(size_out, size_in,dtype=torch.float64)
        self.weights = torch.nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.zeros(size_out,dtype=torch.float64)
        self.bias = torch.nn.Parameter(bias)

    def forward(self,x):
        if (self.size_in>self.size_out):
            # we must truncate the input vector for residual
            xt = torch.narrow(x,-1,0,self.size_out)
        elif(self.size_in<self.size_out):
            xt = torch.nn.functional.pad(x,(self.size_out-self.size_in))
        else:
            xt=x

        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(torch.add(w_times_x, self.bias),xt) 



model = torch.nn.Sequential(
    torch.nn.Linear(1,100,True,dtype=torch.float64),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    ResidualLayer(100,100),
    torch.nn.SiLU(),
    torch.nn.Linear(100,1,True,dtype=torch.float64),
    
)


loss_fn = torch.nn.MSELoss(reduction='mean')

learning_rate = 1E-2
batch_size = 32

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
nbatch = np.ceil(x.shape[0]/(1.0*batch_size))
def do_epoch():
    for batch in range(np.int64(nbatch)):    
        batch_inds = np.arange((batch-1)*batch_size,np.min([batch*batch_size,x.shape[0]]))
        x_batch = x[batch_inds]
        y_pred = model(x_batch)
        loss = torch.mean(torch.mean(torch.square(y_pred-y[batch_inds]),0),0)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

for n_epoch in range(200):
    loss = do_epoch()  
    if np.mod(n_epoch,50)==0:
        print('Loss = ',str(loss.detach().numpy()))

for g in optimizer.param_groups:
    g['lr'] = 1E-3

for n_epoch in range(200):
    loss = do_epoch()  
    if np.mod(n_epoch,50)==0:
        print('Loss = ',str(loss.detach().numpy()))

for g in optimizer.param_groups:
    g['lr'] = 1E-4

for n_epoch in range(200):
    loss = do_epoch()
    if np.mod(n_epoch,50)==0:
        print('Loss = ',str(loss.detach().numpy()))


# make a final prediction

y_pred_final = model(x)

x_np = x.detach().numpy()
y_np = y.detach().numpy()
y_pred_final_np = y_pred_final.detach().numpy()


plot.figure(1)
plot.scatter(x_np,y_np,6,'red')
plot.scatter(x_np,y_pred_final_np,6,'blue')

plot.show()

