import neptune
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# TODO: write new Logger class for TensorFlow v2

class Logger(object):
    def __init__(self, log_dir, neptune_dict = None):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)
        self.neptune_dict = neptune_dict
        if neptune_dict is not None:
            # neptune.init('seutao/sandbox')
            neptune.init(neptune_dict['project'])
            # Create experiment with defined parameters
            neptune.create_experiment(name=neptune_dict['name'],
                                      params=neptune_dict['params'],
                                      upload_source_files=neptune_dict['upload_source_files'])

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
                  value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

        if self.neptune_dict is not None:
            neptune.log_metric(tag, value, timestamp=step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
            tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)

        if self.neptune_dict is not None:
            for tag, value in tag_value_pairs:
                neptune.log_metric(tag, value, timestamp=step)

class DummyLogger:
    def __init__(self, log_dir,neptune_dict = None):
        self.neptune_dict = neptune_dict
        pass

    def scalar_summary(self, tag, value, step):
        pass

    def list_of_scalars_summary(self, tag_value_pairs, step):
        pass


