import tensorflow as tf
import tf_encrypted as tfe

class ModelOwner:

    @tfe.local_computation('model-owner')
    def provide_input(self) -> tf.Tensor:
        # training
        training_data = self._build_data_pipeline()
        weights = self._build_training_graph(training_data)
        return weights

class PredictionClient:

    @tfe.local_computation('prediction-client')
    def provide_input(self) -> tf.Tensor:
        """Prepare input data for prediction."""
        prediction_input, expected_result = self._build_data_pipeline().get_next()
        prediction_input = tf.reshape(
            prediction_input, shape=(self.BATCH_SIZE, ModelOwner.FLATTENED_DIM))
        return prediction_input

    @tfe.local_computation('prediction-client')
    def receive_output(self, logits: tf.Tensor) -> tf.Operation:
        prediction = tf.argmax(logits, axis=1)
        op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
        return op

model_owner = ModelOwner(player_name="model-owner")
prediction_client = PredictionClient(player_name="prediction-client")

# get model weights from model owner
params = model_owner.provide_weights()
# get prediction input from client
x = prediction_client.provide_input()

with tfe.protocol.SecureNN():
    model = tfe.keras.Sequential()
    model.add(tfe.keras.layers.Dense(512, batch_input_shape=x.shape))
    model.add(tfe.keras.layers.Activation('relu'))
    model.add(tfe.keras.layers.Dense(10))
    model.set_weights(params)

    logits = model(x)

# send prediction output back to client
prediction_op = prediction_client.receive_output(logits)

with tfe.Session() as sess:
    sess.run(prediction_op)