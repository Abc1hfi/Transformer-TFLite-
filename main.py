import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
import json
import time

with open('Tokenizers/x_tokenizer.json') as f:
    x_tokenizer_json = json.load(f)

with open('Tokenizers/y_tokenizer.json') as f:
    y_tokenizer_json = json.load(f)

x_tokenizer = tokenizer_from_json(x_tokenizer_json)
y_tokenizer = tokenizer_from_json(y_tokenizer_json)

interpreter = tf.lite.Interpreter(model_path="model.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()


#interpreter.set_tensor(input_details[0]['index'], [sentence, output])

def translate(sentence, max_length=25):
	start = x_tokenizer.texts_to_sequences(["<start>"])[0]
	end = x_tokenizer.texts_to_sequences(["<end>"])[0]

	sentence = "<start> " + sentence + " <end>"
	sentence = x_tokenizer.texts_to_sequences([sentence])
	sentence = pad_sequences(sentence, maxlen=32, padding='post')
	sentence = tf.convert_to_tensor(sentence)

	output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
	output_array.write(0, start).mark_used()

	for i in range(max_length):
		output = tf.transpose(output_array.stack())
		output = pad_sequences(output, maxlen=32)

		interpreter.set_tensor(input_details[0]['index'], sentence)
		interpreter.set_tensor(input_details[1]['index'], output)

		interpreter.invoke()

		predictions = interpreter.get_tensor(output_details[0]['index'])
		predictions = predictions[:, -1:, :]

		predicted_id = np.argmax(predictions)
		print(predicted_id)

		if predicted_id == end:
			break

		output_array = output_array.write(i+1, [predicted_id])

	output = tf.transpose(output_array.stack())

	return output

predicted_sequence = translate("My Friend told me to kill myself")
predicted_sequence = predicted_sequence.numpy()
print(y_tokenizer.sequences_to_texts(predicted_sequence))
