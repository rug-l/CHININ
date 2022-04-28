from IMPORTS import *

class CHININ(torch.nn.Module):
        def __init__(self, D, E, n_met, n_emis, hidden_dims):
            super(CHININ, self).__init__()
            self.nspc  = D.shape[0]
            self.nreac = D.shape[1]
            self.D = torch.tensor(D).to(torch.float32)
            self.E = E
            self.n_met = n_met
            self.n_emis = n_emis

            current_dim = self.nspc + self.nreac + n_met + n_emis
            self.layers = nn.ModuleList()
            for hdim in hidden_dims:
                self.layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            #self.layers.append(nn.Linear(current_dim, self.nreac))
            self.layers.append(nn.Linear(current_dim, self.nspc))

            self.activate = torch.nn.LeakyReLU()
            #self.activate = torch.nn.Tanh()

        def forward(self, x):
            #enc = torch.zeros((self.nreac+self.nspc+self.n_met+self.n_emis), requires_grad=True)
            enc = torch.zeros((self.nreac))

            # E[i] is a list of species indices of the educts in reaction i
            # they get multiplied, as the product is the amount of "resource"
            # for the reaction
            #for i in range(self.nreac):
            #    enc[i] = torch.prod(x[self.E[i]])
            for i in range(self.nreac):
                enc[i] = 1.0
                for j in self.E[i]:
                    enc[i] *= x[j]
            
            # additionally, give single concentrations+met+emis for non-obvious dependencies
            state = torch.cat((enc,x))

            # process (predict "fluxes")
            for layer in self.layers:
                state = self.activate(layer(state))
            
            # convert predicted fluxes to change in concentrations and add previous concentrations
            #out = x[:self.nspc] + torch.matmul(self.D.float(), state)
            #out = x[:self.nspc] + torch.matmul(self.D, state)
            out = x[:self.nspc] + state
            #out=x[:self.nspc]
            #for i in range(self.nspc):
            #    for j in range(self.nreac):
            #        out[i] += D[i,j]*state[j]

            return out

class Feedforward(torch.nn.Module):
        def __init__(self, input_dim, hidden_dims, out_dim):
            super(Feedforward, self).__init__()
            self.input_dim   = input_dim
            self.hidden_dims = hidden_dims
            self.output_dim  = out_dim

            current_dim = input_dim
            self.layers = nn.ModuleList()
            for hdim in hidden_dims:
                self.layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            self.layers.append(nn.Linear(current_dim, self.output_dim))

            self.activate = torch.nn.LeakyReLU()
            #self.activate = torch.nn.Tanh()

        def forward(self, x):
            orig = x[:self.output_dim]
            for layer in self.layers[:-1]:
                x = self.activate(layer(x))
            out = self.layers[-1](x)
            #out = orig + self.layers[-1](x)
            return out

class diurnal_model(torch.nn.Module):
        def __init__(self, core_model):
            super(diurnal_model, self).__init__()
            self.core = core_model

        def forward(self, x, met, emis):
            out=torch.zeros((24,self.core.output_dim))#requires grad?
            for iStep in range(24):
                x = self.core(torch.cat((x, met[iStep,:], emis)))
                out[iStep,:] = x

            return out.flatten()



class ResNet(torch.nn.Module):
        def __init__(self, n_conc, n_met, n_emis, hidden_dims, n_encoded):
            super(ResNet, self).__init__()
            self.n_conc      = n_conc
            self.n_met       = n_met
            self.n_emis      = n_emis
            self.hidden_dims = hidden_dims
            self.n_encoded   = n_encoded

            self.blocks  = nn.ModuleList()
            self.encoder = nn.Linear(n_conc, n_encoded)
            self.decoder = nn.Linear(n_encoded, n_conc)
            for hdim in hidden_sizes:
                self.blocks.append(Feedforward(n_encoded+n_met+n_emis, hdim, n_encoded))

            self.activate = torch.nn.LeakyReLU()
            #self.activate = torch.nn.Tanh()

        def forward(self, x):
            conc_orig, met, emis = x[:self.n_conc], x[self.n_conc:self.n_conc+self.n_met], x[-self.n_emis:]
            conc_orig = self.encoder(conc_orig)
            conc = conc_orig
            out  = conc

            for block in self.blocks:
                conc  = block(torch.cat((conc, met, emis)))
                out  += conc
                conc += conc_orig
            
            out = self.decoder(out)
            return out
