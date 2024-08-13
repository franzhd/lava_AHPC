from lava.proc.lif.process import LIF
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.dense.process import Dense

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import os
import sys
import numpy as np
import typing as ty
from numpy import ndarray

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
sys.path.append(parent_folder)
from AHPC.ahp_compartment import AHPCompartment



# Block structure of the AHPC network
#       +-----------+     +-----------+    +-----------+              +-----------+          +-----------+     +-----------+
# +--+  |           |     |           |    |           |  +--+        |           |          |           |     |           |    +--+
# |i |->|  In_Dense |---->|  In_LIF   |--->|  In_Dense |--| -|------->|  F_LIF    |-------+->| Out_Dense |---->|  O_LIF    |--->|o | 
# +--+  |           |     |           |    |           |  +--+        |           |       |  |           |     |           |    +--+ 
#       +-----------+     +-----------+    +-----------+    A         +-----------+       |  +-----------+     +-----------+
#                                                           |                             |
#                                                           |  +------+ +------+ +------+ |
#                                                           |  |      | |      | |      | |
#                                                           +--|Dense |<| AHPC |<|Dense |<+
#                                                              |      | |      | |      |
#                                                              +------+ +------+ +------+           
#                                       

class AHPCnetwork(AbstractProcess):

    def __init__(self, path, weight_scale=0.0035, **params):

        super().__init__( **params)

        data = np.load(path,allow_pickle=True)

        linear1= data['linear1']
        leaky1_betas= data['leaky1_betas'] 
        leaky1_betas= leaky1_betas if leaky1_betas <= 1 else np.ones(1)
        leaky1_vth= data['leaky1_vth']
        
        print(f"leaky1 betas:{leaky1_betas}")
        print(f"leaky1 vth:{leaky1_vth}")
        linear2= data['linear2']

        leaky2_betas= data['recurrent_betas']
        leaky2_betas= leaky2_betas if  leaky2_betas <= 1 else np.ones(1)

        leaky2_vth= data['recurrent_vth']
        print(f"leaky2 betas:{leaky2_betas}")
        print(f"leaky2 vth:{leaky2_vth}")
        
        linear2= data['linear2'] 
        recurrent_in_weights = data['input_dense']
        recurrent_leaky_betas = data['activation']
        recurrent_leaky_betas= recurrent_leaky_betas if  recurrent_leaky_betas <= 1 else np.ones(1)
        print(f"recurrent betas:{recurrent_leaky_betas}")
        recurrent_out_weights = data['output_dense']
        
        linear3= data['linear3']
        leaky3_betas= data['leaky2_betas']
        leaky3_betas= leaky3_betas if  leaky3_betas <= 1 else np.ones(1)
        leaky3_vth= data['leaky2_vth']
        print(f"leaky3 betas:{leaky3_betas}")
        print(f"leaky3 vth:{leaky3_vth}")
        print(f"recurrent beta:{recurrent_leaky_betas}")
        log_config = params.pop('log_config', 0)
        self.debug = params.pop('debug', False)

           
        #declaration of experiments/net_test.ipynbin and out ports
        self.a_in  = InPort(shape = (linear1.shape[1],))
        self.in_out  = OutPort(shape = (linear1.shape[0],))
        self.f_out = OutPort(shape = (linear2.shape[0],))
        self.s_out  = OutPort(shape = (linear3.shape[0],))

        self.linear_2_out = OutPort(shape = (linear2.shape[0],))

        self.linear1_w = Var(shape=linear1.shape, init=linear1)
        self.leaky1_betas = Var(shape=leaky1_betas.shape, init=(1-leaky1_betas))
        self.leaky1_vth = Var(shape=leaky1_vth.shape, init=leaky1_vth)

        self.linear2_w = Var(shape=linear2.shape, init=linear2)
        self.leaky2_betas = Var(shape=leaky2_betas.shape, init=(1-leaky2_betas))
        self.leaky2_vth = Var(shape=leaky2_vth.shape, init=leaky2_vth)

        self.recurrent_in_w = Var(shape=recurrent_in_weights.shape, init=recurrent_in_weights)
        self.reccurrent_leaky_betas = Var(shape=recurrent_leaky_betas.shape, init=(1-recurrent_leaky_betas))
        self.recurrent_out_w = Var(shape=recurrent_out_weights.shape, init=-recurrent_out_weights)
    
        self.linear3_w = Var(shape=linear3.shape, init=linear3)
        self.leaky3_betas = Var(shape=leaky3_betas.shape, init=(1-leaky3_betas)) 
        self.leaky3_vth = Var(shape=leaky3_vth.shape, init=leaky3_vth)
        
        self.leaky1_v = Var(shape=(self.linear1_w.shape[0],), init=0)
        self.leaky1_u = Var(shape=(self.linear1_w.shape[0],), init=0)

        self.leaky2_v = Var(shape=(self.linear2_w.shape[0],), init=0)
        self.leaky2_u = Var(shape=(self.linear2_w.shape[0],), init=0)

        self.ahpc_v = Var(shape=(self.linear2_w.shape[0],), init=0)
        self.ahpc_u = Var(shape=(self.linear2_w.shape[0],), init=0)

        self.leaky3_v = Var(shape=(self.linear3_w.shape[0],), init=0)
        self.leaky3_u = Var(shape=(self.linear3_w.shape[0],), init=0)

        self.linear1_a_buffer = Var(shape=(self.linear1_w.shape[0],), init=0)
        self.linear2_a_buffer = Var(shape=(self.linear2_w.shape[0],), init=0)
        self.linear3_a_buffer = Var(shape=(self.linear3_w.shape[0],), init=0)
        self.recurrent_in_a_buffer = Var(shape=(self.recurrent_in_w.shape[0],), init=0)
        self.recurrent_out_a_buffer = Var(shape=(self.recurrent_out_w.shape[0],), init=0)

        if self.debug:
            self.input_w = OutPort(shape = (linear1.shape[0],))
            self.f_out  = OutPort(shape = (linear3.shape[1],))
            self.r_current = OutPort(shape = (recurrent_out_weights.shape[0],))
        
        self.log_config = log_config     

    def reset_hidden_state(self):

        self.leaky1_v.set(np.zeros(self.leaky1_v.shape))
        self.leaky1_u.set(np.zeros(self.leaky1_u.shape))

        self.leaky2_v.set(np.zeros(self.leaky2_v.shape))
        self.leaky2_u.set(np.zeros(self.leaky2_u.shape))

        self.ahpc_v.set(np.zeros(self.ahpc_v.shape))
        self.ahpc_u.set(np.zeros(self.ahpc_u.shape))

        self.leaky3_v.set(np.zeros(self.leaky3_v.shape))
        self.leaky3_u.set(np.zeros(self.leaky3_u.shape))

        self.linear1_a_buffer.set(np.zeros(self.linear1_a_buffer.shape))
        self.linear2_a_buffer.set(np.zeros(self.linear2_a_buffer.shape))
        self.linear3_a_buffer.set(np.zeros(self.linear3_a_buffer.shape))

        self.recurrent_in_a_buffer.set(np.zeros(self.recurrent_in_a_buffer.shape))
        self.recurrent_out_a_buffer.set(np.zeros(self.recurrent_out_a_buffer.shape))

    def print_state(self):
        print(f"leaky1_v: {self.leaky1_v.get()}")
        print(f"leaky1_u: {self.leaky1_u.get()}")
        print(f"leaky2_v: {self.leaky2_v.get()}")
        print(f"leaky2_u: {self.leaky2_u.get()}")
        print(f"leaky3_v: {self.leaky3_v.get()}")
        print(f"leaky3_u: {self.leaky3_u.get()}")
        print("")
    
    def print_state_variables(self):
        print('###network state variables###')
        print(f"leaky1_betas: {self.leaky1_betas.get()}")
        print(f"leaky1_vth: {self.leaky1_vth.get()}")
        print(f"leaky2_betas: {self.leaky2_betas.get()}")
        print(f"leaky2_vth: {self.leaky2_vth.get()}")
        print(f"leaky3_betas: {self.leaky3_betas.get()}")
        print(f"leaky3_vth: {self.leaky3_vth.get()}")
        print("")
    
    def print_weights(self):
        print('###network weights###')
        print(f"linear1_w: {self.linear1_w.get()}")
        print(f"linear2_w: {self.linear2_w.get()}")
        print(f"linear3_w: {self.linear3_w.get()}")
        print("")

@implements(proc=AHPCnetwork, protocol=LoihiProtocol)
@requires(CPU)
class PyAHPCnetworkModel(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        # Dense layers has s_in and a_out ports 
        # LIF layers has a_in and s_out ports

    
        self.linear1 = Dense(weights=proc.linear1_w.init, num_message_bits=32)
        
        proc.in_ports.a_in.connect(self.linear1.s_in)

        self.leaky1 = LIF(shape=(proc.linear1_w.shape[0],),
                            u = 0,
                            v = 0,
                            dv = 1,
                            du = proc.leaky1_betas.init,
                            vth=proc.leaky1_vth.init,
                            log_config= proc.log_config,
                            name= "leaky1"
                        )
        self.linear1.a_out.connect(self.leaky1.a_in)
        self.leaky1.s_out.connect(proc.in_out)
        self.linear2 = Dense(weights=proc.linear2_w.init, num_message_bits=32)
        self.linear2.s_in.connect_from(self.leaky1.s_out)

        self.leaky2 = LIF(shape=(proc.linear2_w.shape[0],),
                            u = 0,
                            v = 0,
                            du = 1,
                            dv= proc.leaky2_betas.init,
                            vth=proc.leaky2_vth.init,
                            log_config= proc.log_config,
                            name= "leaky2"
                        )
        self.leaky2.s_out.connect(proc.f_out)
        self.leaky2.a_in.connect_from(self.linear2.a_out)
        
        self.backward = BackwardBranch(in_weight=proc.recurrent_in_w.init,
                                       betas=proc.reccurrent_leaky_betas.init,
                                       out_weight=proc.recurrent_out_w.init)
        
        
        self.leaky2.s_out.connect(self.backward.s_in)

        self.backward.s_out.connect(self.leaky2.a_in)
        #self.backward.s_out.connect(proc.b_out)
        # self.recurrent_out.a_out.connect(self.leaky2.a_in)

        self.linear3 = Dense(weights=proc.linear3_w.init, num_message_bits=0)
        self.linear3.s_in.connect_from(self.leaky2.s_out)

        self.leaky3 = LIF(shape=(proc.linear3_w.shape[0],),
                            u = 0,
                            v = 0,
                            du = 1,
                            dv = proc.leaky3_betas.init,
                            vth=proc.leaky3_vth.init,
                            log_config= proc.log_config,
                            name= "leaky3"
                        )
        self.leaky3.a_in.connect_from(self.linear3.a_out)
        self.leaky3.s_out.connect(proc.s_out)

        print(f"leaky1 dv:{self.leaky1.dv.get()}")
        print(f"leaky1 du:{self.leaky1.du.get()}")
        print(f"leaky1 vth:{self.leaky1.vth.get()}")

        print(f"leaky2 dv:{self.leaky2.dv.get()}")
        print(f"leaky2 du:{self.leaky2.du.get()}")
        print(f"leaky2 vth:{self.leaky2.vth.get()}")

        print(f"recurrent du:{self.backward.betas.get()}")

        print(f"leaky3 dv:{self.leaky3.dv.get()}")
        print(f"leaky3 du:{self.leaky3.du.get()}")
        print(f"leaky3 vth:{self.leaky3.vth.get()}")
        # Connect the processes
        #in connection
        
        
        # self.linear1.a_out.connect(self.leaky1.a_in)

        # self.leaky1.s_out.connect(self.linear2.s_in)        
        # if self.debug:
        #     self.input_w = OutPort(shape = (linear1.shape[0],))
        #     self.f_out  = OutPort(shape = (linear3.shape[1],))
        #     self.r_current = OutPort(shape = (recurrent_out.shape[0],))
        
        # self.log_config = log_config     
        
        # self.leaky2.a_in.connect_from([self.linear2.a_out, self.recurrent_out.a_out])
        
        # self.recurrent_in.a_out.connect(self.ahpc.a_in)
        # self.ahpc.c_out.connect(self.recurrent_out.s_in)
        
        # self.leaky2.s_out.connect([self.recurrent_in.s_in, self.linear3.s_in])
        # self.linear3.a_out.connect(self.leaky3.a_in)
        
        #out connection
        
        
        if proc.debug:
            # self._debug_r_dense = Dense(weights=proc.recurrent_weights.init,
            #         sign_mode = SignMode.INHIBITORY, name="debug_r_dense")
            
            #self.debug_in_dense = Dense(weights=proc.input_weights.init, name="debug_in_dense")
            # self.ahpc.c_out.connect(self._debug_r_dense.s_in)
            #proc.out_ports.input_w.connect_from(self.debug_in_dense.a_out)
            #proc.out_ports.f_out.connect_from(self.f_lif.s_out)
            proc.out_ports.r_current.connect_from(self.backward.s_out)
               
        proc.leaky1_v.alias(self.leaky1.v)
        proc.leaky1_u.alias(self.leaky1.u)

        proc.leaky2_v.alias(self.leaky2.v)
        proc.leaky2_u.alias(self.leaky2.u)

        proc.ahpc_v.alias(self.backward.v)
        proc.ahpc_u.alias(self.backward.u)

        proc.leaky3_v.alias(self.leaky3.v)
        proc.leaky3_u.alias(self.leaky3.u)

        proc.linear1_a_buffer.alias(self.linear1.a_buff)
        proc.linear2_a_buffer.alias(self.linear2.a_buff)
        proc.linear3_a_buffer.alias(self.linear3.a_buff)
        proc.recurrent_in_a_buffer.alias(self.backward.in_buff)
        proc.recurrent_out_a_buffer.alias(self.backward.out_buff)



class BackwardBranch(AbstractProcess):

    def __init__(self, in_weight, betas, out_weight, **params):
        
        super().__init__( **params)
        
        self.s_in  = InPort(shape = (in_weight.shape[1],))
        self.s_out  = OutPort(shape = (out_weight.shape[0],))
        
        self.in_weight = Var(shape=in_weight.shape, init=in_weight)
        self.betas = Var(shape=betas.shape, init=betas)
        self.out_weight = Var(shape=out_weight.shape, init=out_weight)
        #self.linear_2_out = 
        self.in_buff = Var(shape=(in_weight.shape[0],), init=0)
        self.out_buff = Var(shape=(out_weight.shape[0],), init=0)
        self.u = Var(shape=(in_weight.shape[0],), init=0)
        self.v = Var(shape=(in_weight.shape[0],), init=0)
        self.dv = Var(shape=(in_weight.shape[0],), init=1)
        log_config = params.pop('log_config', 0)
        self.log_config = log_config 
        
    def reset_hidden_state(self):

        self.v.set(np.zeros(self.v.shape))
        self.u.set(np.zeros(self.u.shape))
        self.in_buff.set(np.zeros(self.in_buff.shape))
        self.out_buff.set(np.zeros(self.out_buff.shape))
    
    def print_state(self):
        print(f"v: {self.v.get()}")
        print(f"u: {self.u.get()}")
        print("")
    
    def print_state_variables(self):
        print('###recurrent state variables###')
        print(f"betas: {self.betas.get()}")
        print("")

@implements(proc=BackwardBranch, protocol=LoihiProtocol)
@requires(CPU)
class PyBackwardBranch(AbstractSubProcessModel):

    def __init__(self, proc):
        
        self.linear_in = Dense(weights=proc.in_weight.init)
        proc.s_in.connect(self.linear_in.s_in)

        self.leaky = LIF(shape=(proc.in_weight.shape[0],),
                            u = 0,
                            v = 0,
                            du = 1,
                            dv = proc.betas.init,
                            vth=proc.vth.init,
                            log_config= proc.log_config,
                            name= "backward_leaky"
                        )

        self.linear_in.a_out.connect(self.leaky.a_in)

        self.linear_out = Dense(weights=proc.out_weight.init, num_message_bits=32)
        self.linear_out.s_in.connect_from(self.leaky.c_out)
        self.linear_out.a_out.connect(proc.s_out)

        proc.in_buff.alias(self.linear_in.a_buff)
        proc.out_buff.alias(self.linear_out.a_buff)
        proc.u.alias(self.leaky.u)
        proc.v.alias(self.leaky.v)


