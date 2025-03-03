from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat, LearningNeuronModelFixed
)
from lava.proc.lif.models import (
    AbstractPyLifModelFloat, AbstractPyLifModelFixed, AbstractPyLifModelFloat
)
import numpy as np


class LIFEncoder(LIF):
    pass

@implements(proc=LIFEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLifEncoderModelMixed(AbstractPyLifModelFloat):

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: float = LavaPyType(float, float)
    
    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return  self.v > self.vth
         
    
@implements(proc=LIFEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifEncoderModelFloat(AbstractPyLifModelFloat):
    
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth
    

class RSTDPLIF(LearningLIF):
    pass


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in floating point precision with learning enabled
    to do R-STDP.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])
        self.graded_prev = np.zeros(proc_params["shape"]) 

    def spiking_activation(self):
        """Spiking activation function for Learning LIF."""
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: np.ndarray, s_out: np.ndarray) -> np.ndarray:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
         
        reward = np.array(s_out == self.graded_prev).astype(float)
        self.graded_prev = s_graded_in
        return reward

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """
        a_graded_in = self.a_third_factor_in.recv()
        
        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        self.y2 = self.calculate_third_factor_trace(a_graded_in, self.s_out_buff)

        super().run_spk()

        

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class RSTDPLIFBitAcc(LearningNeuronModelFixed, AbstractPyLifModelFixed):
    """Implementation of RSTDP Leaky-Integrate-and-Fire neural
    process bit-accurate with Loihi's hardware LIF dynamics,
    which means, it mimics Loihi behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.s_out_buff = np.zeros(proc_params["shape"])
        self.graded_prev = np.zeros(proc_params["shape"]) 

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth

    def calculate_third_factor_trace(self, s_graded_in: np.ndarray, s_out: np.ndarray) -> np.ndarray:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        reward = np.array(s_out == self.graded_prev).astype(int)
        self.graded_prev = s_graded_in
        return reward

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return np.floor(self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff)

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """
        a_graded_in = self.a_third_factor_in.recv()
        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)
        self.y2 = self.calculate_third_factor_trace(a_graded_in, self.s_out_buff)

        super().run_spk()

        

        

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)
