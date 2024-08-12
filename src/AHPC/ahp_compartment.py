from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.process.neuron import LearningNeuronProcess
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat,
    LearningNeuronModelFixed,
)

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule


from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU

import numpy as np
import typing as ty

from lava.magma.core.process.process import LogConfig




########## ABSTRACT PROCESS DEFINITION ##########

class AHPCompartment(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Union[float, list, np.ndarray],
        v: ty.Union[float, list, np.ndarray],
        du: float,
        dv: float,
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.a_in = InPort(shape=shape)
        self.c_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=u)
        self.v = Var(shape=shape, init=v)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)


class LearningAHPCompartment(LearningNeuronProcess, AHPCompartment):
    """Leaky-Integrate-and-Fire (LIF) neural Process with learning enabled.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    log_config: LogConfig, optional
        Configure the amount of debugging output.
    learning_rule: LearningRule
        Defines the learning parameters and equation.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        learning_rule: Loihi2FLearningRule = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            name=name,
            log_config=log_config,
            learning_rule=learning_rule,
            **kwargs,
        )


########## MODEL ROOT CLASS DEFINITION ##########

class AbstractPyAHPCModelFloat(PyLoihiProcessModel):
    """Abstract implementation of floating point precision
    leaky-integrate-and-fire neuron model.

    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    c_out = None

    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += activation_in
        self.v[:] = self.v * (1 - self.dv) + self.u 

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        
        a_in_data = self.a_in.recv()
        self.subthr_dynamics(activation_in=a_in_data)
        self.c_out.send(self.u)

class AbstractPyAHPCModelFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision
    leaky-integrate-and-fire neuron model. Implementations like those
    bit-accurate with Loihi hardware inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    c_out: None  # This will be an OutPort of different LavaPyTypes
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    du: int = LavaPyType(int, np.uint16, precision=12)
    dv: int = LavaPyType(int, np.uint16, precision=12)

    def __init__(self, proc_params):
        super(AbstractPyAHPCModelFixed, self).__init__(proc_params)
        # ds_offset and dm_offset are 1-bit registers in Loihi 1, which are
        # added to du and dv variables to compute effective decay constants
        # for current and voltage, respectively. They enable setting decay
        # constant values to exact 4096 = 2**12. Without them, the range of
        # 12-bit unsigned du and dv is 0 to 4095.
        self.ds_offset = 1
        self.dm_offset = 0
        self.isthrscaled = False
        # Let's define some bit-widths from Loihi
        # State variables u and v are 24-bits wide
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        # Decays need an MSB alignment with 12-bits
        self.decay_shift = 12
        self.decay_unity = 2**self.decay_shift
        # Threshold and incoming activation are MSB-aligned using 6-bits
        self.act_shift = 6

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """

        # Update current
        # --------------
        decay_const_u = self.du + self.ds_offset
        # Below, u is promoted to int64 to avoid overflow of the product
        # between u and decay constant beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits)
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.sign(decayed_curr) * np.right_shift(
            np.abs(decayed_curr), self.decay_shift
        )
        decayed_curr = np.int32(decayed_curr)
        # Hardware left-shifts synaptic input for MSB alignment
        activation_in = np.left_shift(activation_in, self.act_shift)
        # Add synptic input to decayed current
        decayed_curr += activation_in
        # Check if value of current is within bounds of 24-bit. Overflows are
        # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
        # becomes k and -(2**23 + k) becomes -k
        wrapped_curr = np.where(
            decayed_curr > self.max_uv_val,
            decayed_curr - 2 * self.max_uv_val,
            decayed_curr,
        )
        wrapped_curr = np.where(
            wrapped_curr <= -self.max_uv_val,
            decayed_curr + 2 * self.max_uv_val,
            wrapped_curr,
        )
        self.u[:] = wrapped_curr
        # Update voltage
        # --------------
        decay_const_v = self.dv + self.dm_offset

        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1
        # Decaying voltage similar to current. See the comment above to
        # understand the need for each of the operations below.
        decayed_volt = np.int64(self.v) * (self.decay_unity - decay_const_v)
        decayed_volt = np.sign(decayed_volt) * np.right_shift(
            np.abs(decayed_volt), self.decay_shift
        )
        decayed_volt = np.int32(decayed_volt)
        updated_volt = decayed_volt + self.u
        self.v[:] = np.clip(updated_volt, neg_voltage_limit, pos_voltage_limit)


    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """

        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)
        self.c_out.send(self.u)

########## ACTUAL MODEL DEFINITION ##########
        
@implements(proc=AHPCompartment, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAHPCModelFloat(AbstractPyAHPCModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    c_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=AHPCompartment, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifModelBitAcc(AbstractPyAHPCModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process bit-accurate
    with Loihi's hardware LIF dynamics, which means, it mimics Loihi
    behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - vth: unsigned 17-bit integer (0 to 131071).

    """
    c_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)


@implements(proc=LearningAHPCompartment, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLearningAHPCModelFloat(LearningNeuronModelFloat, AbstractPyAHPCModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision with learning enabled.
    """

    c_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)



@implements(proc=LearningAHPCompartment, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLearningAHPCModelFixed(
    LearningNeuronModelFixed, AbstractPyAHPCModelFixed
):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in fixed point precision with learning enabled.
    """

    c_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        """
        super().run_spk()



