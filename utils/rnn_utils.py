import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

# this is copied from tf 1.4
class ResidualWrapper(tf.contrib.rnn.RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, residual_fn=None):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._residual_fn = residual_fn

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run the cell and then apply the residual_fn on its inputs to its outputs.
    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.
    Returns:
      Tuple of cell outputs and new state.
    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())
    def default_residual_fn(inputs, outputs):
      nest.assert_same_structure(inputs, outputs)
      nest.map_structure(assert_shape_match, inputs, outputs)
      return nest.map_structure(lambda inp, out: inp + out, inputs, outputs)
    res_outputs = (self._residual_fn or default_residual_fn)(inputs, outputs)
    return (res_outputs, new_state)

def single_rnn_cell(cell_name, num_units, train_phase=True, keep_prob=0.75, device_str=None, residual_connection=False, residual_fn=None):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(num_units)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)

    # dropout wrapper
    if train_phase and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob)

    # Residual
    if residual_connection:
        cell = ResidualWrapper(cell, residual_fn = residual_fn)

    # device wrapper
    if device_str:
        print("RNN cell on device:", device_str)
        cell = tf.contrib.rnn.DeviceWrapper(cell, device_str)
    return cell


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def get_cell_list(cell_name, num_units, num_layers, num_residual_layers=0,
                  train_phase=True, num_gpus=1, base_gpu=0, keep_prob=0.8, residual_fn=None):
    """Create a list of RNN cells."""
    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        single_cell = single_rnn_cell(
            cell_name=cell_name,
            num_units=num_units,
            keep_prob=keep_prob,
            train_phase=train_phase,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
            residual_fn=residual_fn
        )
        cell_list.append(single_cell)

    return cell_list

def multi_rnn_cell(cell_name, dim_size, num_layers=1, train_phase=True, keep_prob=0.80, num_residual_layers=0, num_gpus=1, base_gpu=0, residual_fn=None):
    """
    Get multi layer rnn cell
    """
    cells = get_cell_list(cell_name, dim_size, num_layers,
                          num_residual_layers, train_phase, 
                          num_gpus, base_gpu, keep_prob, residual_fn)
    if len(cells) > 1:
        final_cell = tf.contrib.rnn.MultiRNNCell(cells=cells)
    else:
        final_cell = cells[0]
    return final_cell

    
def bidirection_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, num_gpus, sequence_length, inputs):
    fw_cell = multi_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, 0, num_gpus)
    bw_cell = multi_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, 0, num_gpus, num_bi_layers)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        dtype=tf.float32,
        sequence_length=sequence_length,
        inputs=inputs,
        swap_memory = True
    )
    outputs_concat = tf.concat(outputs, -1)
    return outputs_concat, states