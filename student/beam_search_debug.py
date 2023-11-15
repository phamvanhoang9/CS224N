import numpy as np

def beam_search_decoder(predictions, beam_width):
  sequences = [(list(), 1.0)]

  for prediction in predictions:
    all_candidates = list()
    for seq, score in sequences:
      for i in range(len(prediction)):
        candidate = [seq + [i], score * prediction[i]]
        all_candidates.append(candidate)

    ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    sequences = ordered[:beam_width]

  return sequences

predictions = np.array([
    [0.1, 0.3, 0.6],
    [0.4, 0.2, 0.4],
    [0.2, 0.5, 0.3],
])

beam_width = 2
result = beam_search_decoder(predictions, beam_width)

for seq, prob in result:
  print(f"Sequence: {seq}, Probability: {prob}")