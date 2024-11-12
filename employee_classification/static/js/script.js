function makePrediction() {
                              const form = document.getElementById('predictionForm');
                              const formData = new FormData(form);
                          
                              fetch('/predict', {
                                  method: 'POST',
                                  body: formData
                              })
                              .then(response => response.json())
                              .then(data => {
                                  document.getElementById('result').textContent = "Prediction: " + data.prediction;
                              })
                              .catch(error => console.error('Error:', error));
                          }
                          